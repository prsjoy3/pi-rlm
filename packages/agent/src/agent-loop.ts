/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@earendil-works/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	StreamFn,
} from "./types.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	initialContext: AgentContext,
	newMessages: AgentMessage[],
	initialConfig: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<void> {
	let currentContext = initialContext;
	let config = initialConfig;
	let firstRun = true;

	while (true) {
		if (!firstRun) {
			await emit({ type: "turn_start" });
		}
		firstRun = false;

		const pendingMessages = (await config.getSteeringMessages?.()) || [];
		for (const message of pendingMessages) {
			await emit({ type: "message_start", message });
			await emit({ type: "message_end", message });
			currentContext.messages.push(message);
			newMessages.push(message);
		}

		const result = await runRlmTurn(currentContext, newMessages, config, signal, emit, streamFn);
		currentContext = result.context;
		config = result.config;

		await emit({ type: "turn_end", message: result.message, toolResults: result.toolResults });

		if (
			await config.shouldStopAfterTurn?.({
				message: result.message,
				toolResults: result.toolResults,
				context: currentContext,
				newMessages,
			})
		) {
			await emit({ type: "agent_end", messages: newMessages });
			return;
		}

		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length === 0) {
			await emit({ type: "agent_end", messages: newMessages });
			return;
		}
		for (const message of followUpMessages) {
			await emit({ type: "message_start", message });
			await emit({ type: "message_end", message });
			currentContext.messages.push(message);
			newMessages.push(message);
		}
	}
}

const RLM_MAX_ITERATIONS = 12;
const RLM_MAX_OUTPUT_CHARS = 10_000;
const RLM_ACTION_PROMPT = `You are the recursive language model core for pi-rlm.

You have an iterative runtime. At each step, return exactly one JSON object and no markdown.
State persists for this agent turn through the RLM history shown below. Use small steps: inspect with tools, observe output, then decide the next step.

Actions:
{"action":"tool","tool":"read","args":{"path":"package.json"},"reasoning":"why this tool call is next"}
{"action":"llm_query","prompt":"focused semantic question over known context","reasoning":"why a sub-query helps"}
{"action":"submit","answer":"final user-facing response","reasoning":"why the task is complete"}

Rules:
- Use available tools by name when you need workspace information or file changes.
- Do not invent tool results; wait for observations.
- Call submit only when you have enough evidence or the task is impossible.
- If a tool fails, inspect the observation and recover in the next step.`;

type RlmAction =
	| { action: "tool"; tool: string; args?: unknown; reasoning?: string }
	| { action: "llm_query"; prompt: string; reasoning?: string }
	| { action: "submit"; answer: string; reasoning?: string };

type RlmHistoryEntry = {
	reasoning: string;
	action: string;
	observation: string;
};

type RlmTurnResult = {
	context: AgentContext;
	config: AgentLoopConfig;
	message: AssistantMessage;
	toolResults: ToolResultMessage[];
};

async function runRlmTurn(
	initialContext: AgentContext,
	newMessages: AgentMessage[],
	initialConfig: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<RlmTurnResult> {
	const currentContext = initialContext;
	const config = initialConfig;
	const history: RlmHistoryEntry[] = [];
	const toolResults: ToolResultMessage[] = [];

	for (let iteration = 0; iteration < RLM_MAX_ITERATIONS; iteration++) {
		if (signal?.aborted) {
			const message = createAssistantMessage("Operation aborted", config, "aborted", "Operation aborted");
			await emitAssistantMessage(message, currentContext, newMessages, emit);
			return { context: currentContext, config, message, toolResults };
		}

		const actionText = await generateRlmAction(currentContext, config, history, iteration, signal, streamFn);
		const action = parseRlmAction(actionText);
		if (!action) {
			await emitAssistantMessage(
				createAssistantMessage(
					`RLM step ${iteration + 1}: invalid action\n\nThe model did not return valid RLM JSON. Asking it to try again.`,
					config,
					"stop",
				),
				currentContext,
				newMessages,
				emit,
			);
			history.push({
				reasoning: "The model did not return valid RLM JSON.",
				action: actionText,
				observation: "Return exactly one JSON object with action tool, llm_query, or submit.",
			});
			continue;
		}

		if (action.action === "submit") {
			const message = createAssistantMessage(action.answer, config, "stop");
			await emitAssistantMessage(message, currentContext, newMessages, emit);
			return await prepareRlmNextTurn(currentContext, config, message, toolResults, newMessages);
		}

		if (action.action === "llm_query") {
			await emitAssistantMessage(
				createAssistantMessage(formatRlmActionTrace(iteration, action), config, "stop"),
				currentContext,
				newMessages,
				emit,
			);
			const observation = await runRlmSubQuery(currentContext, config, action.prompt, signal, streamFn);
			history.push({
				reasoning: action.reasoning ?? "",
				action: JSON.stringify(action),
				observation: truncateRlmOutput(observation),
			});
			continue;
		}

		const syntheticAssistant = createToolCallAssistantMessage(action, config, iteration);
		await emitAssistantMessage(syntheticAssistant, currentContext, newMessages, emit);
		const executedToolBatch = await executeToolCalls(currentContext, syntheticAssistant, config, signal, emit);
		for (const result of executedToolBatch.messages) {
			currentContext.messages.push(result);
			newMessages.push(result);
			toolResults.push(result);
		}
		history.push({
			reasoning: action.reasoning ?? "",
			action: JSON.stringify(action),
			observation: truncateRlmOutput(formatToolResultsForRlm(executedToolBatch.messages)),
		});

		if (executedToolBatch.terminate) {
			const message = createAssistantMessage("Task completed.", config, "stop");
			await emitAssistantMessage(message, currentContext, newMessages, emit);
			return await prepareRlmNextTurn(currentContext, config, message, toolResults, newMessages);
		}
	}

	const fallback = await generateRlmFallbackAnswer(currentContext, config, history, signal, streamFn);
	const message = createAssistantMessage(fallback, config, "stop");
	await emitAssistantMessage(message, currentContext, newMessages, emit);
	return await prepareRlmNextTurn(currentContext, config, message, toolResults, newMessages);
}

async function prepareRlmNextTurn(
	currentContext: AgentContext,
	config: AgentLoopConfig,
	message: AssistantMessage,
	toolResults: ToolResultMessage[],
	newMessages: AgentMessage[],
): Promise<RlmTurnResult> {
	const nextTurnSnapshot = await config.prepareNextTurn?.({
		message,
		toolResults,
		context: currentContext,
		newMessages,
	});
	if (!nextTurnSnapshot) {
		return { context: currentContext, config, message, toolResults };
	}
	return {
		context: nextTurnSnapshot.context ?? currentContext,
		config: {
			...config,
			model: nextTurnSnapshot.model ?? config.model,
			reasoning:
				nextTurnSnapshot.thinkingLevel === undefined
					? config.reasoning
					: nextTurnSnapshot.thinkingLevel === "off"
						? undefined
						: nextTurnSnapshot.thinkingLevel,
		},
		message,
		toolResults,
	};
}

async function generateRlmAction(
	context: AgentContext,
	config: AgentLoopConfig,
	history: RlmHistoryEntry[],
	iteration: number,
	signal: AbortSignal | undefined,
	streamFn?: StreamFn,
): Promise<string> {
	return runHiddenAssistantText(
		context,
		config,
		buildRlmPrompt(context, config, history, iteration),
		signal,
		streamFn,
	);
}

async function runRlmSubQuery(
	context: AgentContext,
	config: AgentLoopConfig,
	prompt: string,
	signal: AbortSignal | undefined,
	streamFn?: StreamFn,
): Promise<string> {
	return runHiddenAssistantText(
		context,
		config,
		`Answer this focused sub-query using the visible conversation context. Return only the answer.\n\n${prompt}`,
		signal,
		streamFn,
	);
}

async function generateRlmFallbackAnswer(
	context: AgentContext,
	config: AgentLoopConfig,
	history: RlmHistoryEntry[],
	signal: AbortSignal | undefined,
	streamFn?: StreamFn,
): Promise<string> {
	return runHiddenAssistantText(
		context,
		config,
		`The RLM loop reached its iteration limit. Based on the trajectory, write the best final user-facing answer now.\n\n${formatRlmHistory(history)}`,
		signal,
		streamFn,
	);
}

function buildRlmPrompt(
	context: AgentContext,
	config: AgentLoopConfig,
	history: RlmHistoryEntry[],
	iteration: number,
): string {
	const toolDocs = (context.tools ?? []).map((tool) => `- ${tool.name}: ${tool.description}`).join("\n");
	return `${RLM_ACTION_PROMPT}\n\nIteration: ${iteration + 1}/${RLM_MAX_ITERATIONS}\nModel: ${config.model.provider}/${config.model.id}\nAvailable tools:\n${toolDocs || "(none)"}\n\nRLM history:\n${formatRlmHistory(history)}\n\nReturn the next JSON action.`;
}

function formatRlmHistory(history: RlmHistoryEntry[]): string {
	if (history.length === 0) {
		return "No RLM steps yet.";
	}
	return history
		.map(
			(entry, index) =>
				`Step ${index + 1}\nReasoning: ${entry.reasoning}\nAction: ${entry.action}\nObservation:\n${entry.observation}`,
		)
		.join("\n\n");
}

async function runHiddenAssistantText(
	context: AgentContext,
	config: AgentLoopConfig,
	prompt: string,
	signal: AbortSignal | undefined,
	streamFn?: StreamFn,
): Promise<string> {
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}
	const llmMessages = await config.convertToLlm(messages);
	const hiddenPrompt = { role: "user", content: prompt, timestamp: Date.now() } as const;
	const streamFunction = streamFn || streamSimple;
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;
	const response = await streamFunction(
		config.model,
		{
			systemPrompt: context.systemPrompt,
			messages: [...llmMessages, hiddenPrompt],
		},
		{ ...config, apiKey: resolvedApiKey, signal },
	);
	for await (const _event of response) {
		// Drain stream; final text is read from response.result().
	}
	return assistantText(await response.result());
}

function parseRlmAction(text: string): RlmAction | undefined {
	const stripped = stripJsonFences(text);
	const jsonText = extractJsonObject(stripped);
	if (!jsonText) {
		return undefined;
	}
	try {
		const value = JSON.parse(jsonText) as unknown;
		if (!isRecord(value) || typeof value.action !== "string") {
			return undefined;
		}
		if (value.action === "submit" && typeof value.answer === "string") {
			return { action: "submit", answer: value.answer, reasoning: optionalString(value.reasoning) };
		}
		if (value.action === "llm_query" && typeof value.prompt === "string") {
			return { action: "llm_query", prompt: value.prompt, reasoning: optionalString(value.reasoning) };
		}
		if (value.action === "tool" && typeof value.tool === "string") {
			return { action: "tool", tool: value.tool, args: value.args, reasoning: optionalString(value.reasoning) };
		}
		return undefined;
	} catch {
		return undefined;
	}
}

function stripJsonFences(text: string): string {
	const trimmed = text.trim();
	if (!trimmed.startsWith("```")) {
		return trimmed;
	}
	return trimmed
		.replace(/^```(?:json)?\s*/i, "")
		.replace(/```$/i, "")
		.trim();
}

function extractJsonObject(text: string): string | undefined {
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	if (start === -1 || end === -1 || end <= start) {
		return undefined;
	}
	return text.slice(start, end + 1);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function createToolCallAssistantMessage(
	action: Extract<RlmAction, { action: "tool" }>,
	config: AgentLoopConfig,
	iteration: number,
): AssistantMessage {
	const args = isRecord(action.args) ? action.args : {};
	return {
		...createAssistantMessage("", config, "toolUse"),
		content: [
			{ type: "text", text: formatRlmActionTrace(iteration, action) },
			{
				type: "toolCall",
				id: `rlm_${Date.now()}_${Math.random().toString(36).slice(2)}`,
				name: action.tool,
				arguments: args,
			},
		],
	};
}

function formatRlmActionTrace(iteration: number, action: RlmAction): string {
	if (action.action === "submit") {
		return `RLM step ${iteration + 1}: submit final answer\n\nReasoning: ${action.reasoning ?? ""}`;
	}
	if (action.action === "llm_query") {
		return `RLM step ${iteration + 1}: llm_query\n\nReasoning: ${action.reasoning ?? ""}\n\nPrompt:\n${action.prompt}`;
	}
	return `RLM step ${iteration + 1}: tool ${action.tool}\n\nReasoning: ${action.reasoning ?? ""}\n\nArguments:\n${JSON.stringify(action.args ?? {}, null, 2)}`;
}

function createAssistantMessage(
	text: string,
	config: AgentLoopConfig,
	stopReason: AssistantMessage["stopReason"],
	errorMessage?: string,
): AssistantMessage {
	return {
		role: "assistant",
		content: text ? [{ type: "text", text }] : [],
		api: config.model.api,
		provider: config.model.provider,
		model: config.model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		errorMessage,
		timestamp: Date.now(),
	};
}

async function emitAssistantMessage(
	message: AssistantMessage,
	context: AgentContext,
	newMessages: AgentMessage[],
	emit: AgentEventSink,
): Promise<void> {
	context.messages.push(message);
	newMessages.push(message);
	await emit({ type: "message_start", message: { ...message } });
	if (message.content.some((content) => content.type === "text")) {
		await emit({
			type: "message_update",
			message: { ...message },
			assistantMessageEvent: { type: "text_start", contentIndex: 0, partial: message },
		});
		const text = assistantText(message);
		await emit({
			type: "message_update",
			message: { ...message },
			assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text, partial: message },
		});
		await emit({
			type: "message_update",
			message: { ...message },
			assistantMessageEvent: { type: "text_end", contentIndex: 0, content: text, partial: message },
		});
	}
	await emit({ type: "message_end", message });
}

function assistantText(message: AssistantMessage): string {
	return message.content
		.filter((content) => content.type === "text")
		.map((content) => content.text)
		.join("");
}

function formatToolResultsForRlm(messages: ToolResultMessage[]): string {
	return messages
		.map((message) => {
			const content = message.content
				.map((item) => (item.type === "text" ? item.text : `[image: ${item.mimeType}]`))
				.join("\n");
			return `${message.toolName} ${message.isError ? "errored" : "returned"}:\n${content}`;
		})
		.join("\n\n");
}

function truncateRlmOutput(output: string): string {
	if (output.length <= RLM_MAX_OUTPUT_CHARS) {
		return output;
	}
	const half = Math.floor(RLM_MAX_OUTPUT_CHARS / 2);
	return `${output.slice(0, half)}\n\n... (${output.length - RLM_MAX_OUTPUT_CHARS} characters omitted) ...\n\n${output.slice(-half)}`;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const hasSequentialToolCall = toolCalls.some(
		(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
	);
	if (config.toolExecution === "sequential" || hasSequentialToolCall) {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

type ExecutedToolCallBatch = {
	messages: ToolResultMessage[];
	terminate: boolean;
};

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallOutcome[] = [];
	const messages: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		let finalized: FinalizedToolCallOutcome;
		if (preparation.kind === "immediate") {
			finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			};
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
		}

		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		finalizedCalls.push(finalized);
		messages.push(toolResultMessage);

		if (signal?.aborted) {
			break;
		}
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	};
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			} satisfies FinalizedToolCallOutcome;
			await emitToolExecutionEnd(finalized, emit);
			finalizedCalls.push(finalized);
			if (signal?.aborted) {
				break;
			}
			continue;
		}

		finalizedCalls.push(async () => {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return finalized;
		});
		if (signal?.aborted) {
			break;
		}
	}

	const orderedFinalizedCalls = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
	const messages: ToolResultMessage[] = [];
	for (const finalized of orderedFinalizedCalls) {
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(orderedFinalizedCalls),
	};
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallEntry = FinalizedToolCallOutcome | (() => Promise<FinalizedToolCallOutcome>);

function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
	return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (signal?.aborted) {
				return {
					kind: "immediate",
					result: createErrorToolResult("Operation aborted"),
					isError: true,
				};
			}
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}
		if (signal?.aborted) {
			return {
				kind: "immediate",
				result: createErrorToolResult("Operation aborted"),
				isError: true,
			};
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
	};
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		content: finalized.result.content,
		details: finalized.result.details,
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
