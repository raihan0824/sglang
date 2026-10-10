import json
import logging
import re
from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.call_conformance import log_call_conformance
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.param_coercion import OMIT, ToolParamSchema

logger = logging.getLogger(__name__)

# Stable grep prefix for the per-call conformance warnings.
LOG_PREFIX = "qwen3_coder tool call"


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None
        self.current_param_schema: ToolParamSchema = ToolParamSchema(None)
        # Arguments streamed for the open call and the request's tools, for the
        # conformance check when the call closes.
        self._streamed_args: str = ""
        self._tools: Optional[List[Tool]] = None

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    @staticmethod
    def _normalize_func_name(raw_name: str, tools: Optional[list[Tool]]) -> str:
        name = raw_name.strip()
        names = {tool.function.name for tool in tools or []}
        if name not in names and name.startswith("functions."):
            unprefixed = name[len("functions.") :]
            if unprefixed in names:
                return unprefixed
        return name

    def _get_param_schema(
        self, func_name: Optional[str], tools: Optional[list[Tool]]
    ) -> ToolParamSchema:
        for tool in tools or []:
            if tool.type == "function" and tool.function.name == func_name:
                return ToolParamSchema(tool.function.parameters)
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return ToolParamSchema(None)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            raw_tool_calls = self.tool_call_regex.findall(text)
            if not raw_tool_calls:
                # Fallback: maybe the whole text is inside the tag or tags are stripped
                if self.tool_call_prefix in text:
                    raw_tool_calls = [text]

            tool_idx = 0
            for tool_content in raw_tool_calls:
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = self._normalize_func_name(func_body[:name_end], tools)
                    params_str = func_body[name_end + 1 :]

                    param_schema = self._get_param_schema(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx].strip()
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        value = param_schema.convert(p_name, p_val)
                        if value is not OMIT:
                            parsed_params[p_name] = value

                    arguments = json.dumps(parsed_params, ensure_ascii=False)
                    log_call_conformance(
                        LOG_PREFIX, func_name, arguments, tools, "nonstream"
                    )
                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx, name=func_name, parameters=arguments
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser.
        """
        self._buffer += new_text
        self._tools = tools

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = self._normalize_func_name(
                        current_slice[len(self.tool_call_prefix) : end_angle], tools
                    )
                    self._close_open_function(calls)

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name
                    self._streamed_args = ""
                    self.current_param_schema = self._get_param_schema(func_name, tools)

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_angle + 1
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if (
                current_slice.startswith(self.parameter_prefix)
                and self.current_func_name is not None
            ):
                name_end = current_slice.find(">")
                if name_end != -1:
                    value_start_idx = name_end + 1
                    rest_of_slice = current_slice[value_start_idx:]

                    # A parameter can end in multiple ways:
                    # 1. [Normal] Encounter </parameter>
                    # 2. [Abnormal] Encounter next <parameter=
                    # 3. [Abnormal] Encounter </function>
                    # So we need to find the smallest one as the parameter end position.
                    cand_end_param = rest_of_slice.find(self.parameter_end_token)
                    cand_next_param = rest_of_slice.find(self.parameter_prefix)
                    cand_end_func = rest_of_slice.find(self.function_end_token)

                    candidates = []
                    if cand_end_param != -1:
                        candidates.append(
                            (cand_end_param, len(self.parameter_end_token))
                        )
                    if cand_next_param != -1:
                        candidates.append((cand_next_param, 0))
                    if cand_end_func != -1:
                        candidates.append((cand_end_func, 0))

                    if candidates:
                        best_cand = min(candidates, key=lambda x: x[0])
                        end_pos = best_cand[0]
                        end_token_len = best_cand[1]

                        param_name = current_slice[
                            len(self.parameter_prefix) : name_end
                        ].strip()
                        raw_value = rest_of_slice[:end_pos]

                        # Cleanup value
                        if raw_value.startswith("\n"):
                            raw_value = raw_value[1:]
                        if raw_value.endswith("\n"):
                            raw_value = raw_value[:-1]

                        self._emit_param(calls, param_name, raw_value)

                        # Advance cursor
                        total_len = (name_end + 1) + end_pos + end_token_len
                        self.parsed_pos += total_len
                        continue

                # Incomplete parameter tag or value
                break

            # -------------------------------------------------------
            # 4. Function End: </function>
            # -------------------------------------------------------
            if (
                current_slice.startswith(self.function_end_token)
                and self.current_func_name is not None
            ):
                self._close_open_function(calls)
                self.parsed_pos += len(self.function_end_token)
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self._close_open_function(calls)
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(current_slice)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self.parsed_pos += len(current_slice)
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]

                is_potential_tag = False
                for tag in possible_tags:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol
                    if not self.is_inside_tool_call:
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(text_segment)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self.parsed_pos += next_open_angle
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def _emit_param(self, calls: list, param_name: str, raw_value: str) -> None:
        if raw_value.startswith("\n"):
            raw_value = raw_value[1:]
        if raw_value.endswith("\n"):
            raw_value = raw_value[:-1]
        converted_val = self.current_param_schema.convert(param_name, raw_value)
        if converted_val is OMIT:
            return
        if not self.json_started:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self._streamed_args += "{"
            self.json_started = True
        json_key_val = (
            f"{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}"
        )
        if self.current_tool_param_count > 0:
            json_key_val = f", {json_key_val}"
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, parameters=json_key_val)
        )
        self._streamed_args += json_key_val
        self.current_tool_param_count += 1

    def _close_open_function(self, calls: list, mode: str = "stream") -> None:
        """Emit the closing brace of the function being streamed, if any."""
        if self.current_func_name is None:
            return
        if not self.json_started:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self._streamed_args += "{"
            self.json_started = True
        calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="}"))
        self._streamed_args += "}"
        log_call_conformance(
            LOG_PREFIX, self.current_func_name, self._streamed_args, self._tools, mode
        )
        self.current_func_name = None

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Close a call the stream ended inside, as detect_and_parse would parse it."""
        calls = []
        rest = self._buffer[self.parsed_pos :]
        self._buffer, self.parsed_pos = "", 0
        self._tools = tools
        if self.current_func_name is not None:
            if rest.startswith(self.parameter_prefix) and ">" in rest:
                name_end = rest.index(">")
                param_name = rest[len(self.parameter_prefix) : name_end].strip()
                self._emit_param(calls, param_name, rest[name_end + 1 :])
            self._close_open_function(calls, mode="stream-eos")
            return StreamingParseResult(calls=calls)
        if self.is_inside_tool_call:
            return StreamingParseResult()
        return StreamingParseResult(normal_text=rest)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ):
        # xgrammar's qwen_xml grammar for a non-strict tool spells parameter
        # names as identifiers only, so a declared "-i" or "a.b" cannot be
        # written; constrain forced calls by the declared schema instead.
        if tool_choice != "auto" and tools:
            tools = [
                tool.model_copy(
                    update={
                        "function": tool.function.model_copy(update={"strict": True})
                    }
                )
                for tool in tools
            ]
        return super().get_structural_tag(
            tools=tools,
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )
