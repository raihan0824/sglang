from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV41Detector(DeepSeekV32Detector):
    """DeepSeek V4.1 DSML detector.

    The leading space in each tag name below is intentional, not a typo.
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"
    strip_string_param_value: bool = False

    # The encoder joins an assistant turn's content and its calls block with a
    # blank line, and renders it even when there is no content.
    tool_calls_prefix = "\n\n"
    think_end_token = "</think>"

    def get_structural_tag_name(self) -> Optional[str]:
        return "deepseek_v4_1"

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ):
        # A forced call always follows the declared schema. Gateways drop
        # "strict" on the way here (LiteLLM's hosted_vllm adapter does), and a
        # free JSON body lets the model close a forced call with "{}".
        if tools and (tool_choice == "required" or isinstance(tool_choice, ToolChoice)):
            tools = [
                tool.model_copy(
                    update={"function": tool.function.model_copy(update={"strict": True})}
                )
                for tool in tools
            ]
        return super().get_structural_tag(
            tools=tools,
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )
