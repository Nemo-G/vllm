# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Union

import json
import regex as re

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("deepseek_v3_1")
class DeepSeekV3_1ToolParser(ToolParser):
    """
    Tool parser for DeepSeek-V3.1 model using the official format:
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function_name<｜tool▁sep｜>arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = (
            [])  # map what has been streamed for each tool so far to a list
        self.json_buffer: str = ""  # Buffer for accumulating JSON content
        self.in_json_tool_call: bool = False  # Track if we're in a JSON tool call
        self.xml_buffer: str = ""  # Buffer for accumulating XML content
        self.in_xml_tool_call: bool = False  # Track if we're in an XML tool call
        self.xml_function_name: str = ""  # Current function name being streamed
        self.xml_current_param: str = ""  # Current parameter being built
        self.xml_current_param_name: str = ""  # Name of current parameter
        self.xml_params: dict = {}  # Accumulated parameters so far

        # Official DeepSeek-V3.1 tokens
        self.tool_calls_start_token: str = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token: str = "<｜tool▁calls▁end｜>"

        self.tool_call_start_token: str = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token: str = "<｜tool▁call▁end｜>"
        self.tool_sep_token: str = "<｜tool▁sep｜>"

        # Official format: <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>arguments<｜tool▁call▁end｜>
        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>[^<]+)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>",
            re.DOTALL
        )

        # For streaming: partial matches
        self.stream_tool_call_portion_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>[^<]+)<｜tool▁sep｜>(?P<function_arguments>.*)",
            re.DOTALL
        )

        self.stream_tool_call_name_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>[^<]+)<｜tool▁sep｜>"
        )

        # Patterns for detecting JSON tool calls in content
        self.json_tool_call_start_regex = re.compile(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{')
        self.json_tool_call_complete_regex = re.compile(
            r'\{"name":\s*"(?P<function_name>[^"]+)",\s*"arguments":\s*(?P<function_arguments>\{.*?\})\s*\}',
            re.DOTALL
        )

        # Patterns for detecting XML-style function calls  
        self.xml_function_calls_start_regex = re.compile(r'<function_calls>')
        self.xml_function_calls_complete_regex = re.compile(
            r'<function_calls>\s*<invoke\s+name="(?P<function_name>[^"]+)">\s*(?P<parameters>.*?)\s*</invoke>\s*</function_calls>',
            re.DOTALL
        )
        self.xml_parameter_regex = re.compile(r'<parameter\s+name="([^"]+)">([^<]*)</parameter>', re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        
        # Get token IDs
        self.tool_calls_start_token_id = self.vocab.get(
            self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(
            self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (self.tool_calls_start_token_id is None
                or self.tool_calls_end_token_id is None):
            raise RuntimeError(
                "DeepSeek-V3.1 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

    def _reset_streaming_state(self):
        """Reset streaming state for clean transitions between formats."""
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.json_buffer = ""
        self.in_json_tool_call = False
        self.xml_buffer = ""
        self.in_xml_tool_call = False
        self.xml_function_name = ""
        self.xml_current_param = ""
        self.xml_current_param_name = ""
        self.xml_params = {}

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # Check for official DeepSeek format first
        if self.tool_calls_start_token in model_output:
            try:
                # Find all tool calls using the regex pattern
                function_call_tuples = self.tool_call_regex.findall(
                    model_output)

                tool_calls = []
                for match in function_call_tuples:
                    function_name, function_args = match
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(name=function_name,
                                                  arguments=function_args),
                        ))

                content = model_output[:model_output.
                                       find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception(
                    "Error in extracting tool call from response.")
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

        # Check for JSON format tool calls
        json_matches = self.json_tool_call_complete_regex.findall(model_output)
        if json_matches:
            try:
                tool_calls = []
                for match in json_matches:
                    function_name, function_args = match
                    # Parse the JSON arguments to ensure they're valid
                    parsed_args = json.loads(function_args)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(name=function_name,
                                                  arguments=json.dumps(parsed_args)),
                        ))

                # Extract content before the first JSON tool call
                first_json_match = self.json_tool_call_complete_regex.search(model_output)
                if first_json_match:
                    content = model_output[:first_json_match.start()].rstrip()
                else:
                    content = model_output

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception as e:
                logger.exception(
                    "Error in extracting JSON tool call from response: %s", e)
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

        # Check for XML format tool calls
        xml_matches = self.xml_function_calls_complete_regex.findall(model_output)
        if xml_matches:
            try:
                tool_calls = []
                for match in xml_matches:
                    function_name, parameters_section = match
                    
                    # Parse parameters from XML
                    params = {}
                    param_matches = self.xml_parameter_regex.findall(parameters_section)
                    for param_name, param_value in param_matches:
                        params[param_name] = param_value.strip()
                    
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(name=function_name,
                                                  arguments=json.dumps(params)),
                        ))

                # Extract content before the first XML tool call
                first_xml_match = self.xml_function_calls_complete_regex.search(model_output)
                if first_xml_match:
                    content = model_output[:first_xml_match.start()].rstrip()
                else:
                    content = model_output

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception as e:
                logger.exception(
                    "Error in extracting XML tool call from response: %s", e)
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

        # No tool calls found
        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        
        # Handle EOS tokens and finish_reason logic
        if not delta_text:
            # Check if this is an EOS token after tool calls are complete
            if (delta_token_ids and len(self.prev_tool_call_arr) > 0):
                # We have completed tool calls, check if all are properly closed
                if (not self.in_xml_tool_call and not self.in_json_tool_call):
                    # All tool calls are complete, return empty delta to allow finish_reason processing
                    return DeltaMessage(content="")
                elif not self.in_xml_tool_call and not self.in_json_tool_call and not current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        try:
            # First, check for official DeepSeek format (with special tokens)
            if self.tool_calls_start_token_id in current_token_ids:
                return self._handle_official_format_streaming(
                    previous_text, current_text, delta_text, 
                    previous_token_ids, current_token_ids, delta_token_ids
                )
            
            # Check for XML format tool calls (only if we detect the pattern)
            if '<function_calls>' in current_text and not self.in_json_tool_call:
                return self._handle_xml_format_streaming(
                    previous_text, current_text, delta_text
                )
            
            # Continue XML processing if already in XML mode
            if self.in_xml_tool_call:
                return self._handle_xml_format_streaming(
                    previous_text, current_text, delta_text
                )
            
            # Handle JSON format tool calls in content stream
            return self._handle_json_format_streaming(
                previous_text, current_text, delta_text
            )

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.

    def _handle_official_format_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """Handle the official DeepSeek format with special tokens."""
            
        # Remove tool call markers from delta text for cleaner output
        delta_text = delta_text.replace(self.tool_calls_start_token,
                                        "").replace(self.tool_calls_end_token,
                                                    "")

        # figure out where we are in the parsing by counting tool call
        # start & end tags
        prev_tool_start_count = previous_token_ids.count(
            self.tool_call_start_token_id)
        prev_tool_end_count = previous_token_ids.count(
            self.tool_call_end_token_id)
        cur_tool_start_count = current_token_ids.count(
            self.tool_call_start_token_id)
        cur_tool_end_count = current_token_ids.count(
            self.tool_call_end_token_id)
        tool_call_portion = None
        text_portion = None

        # case: if we're generating text, OR rounding out a tool call
        if (cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text):
            logger.debug("Generating text content! skipping tool parsing.")
            return DeltaMessage(content=delta_text)

        if self.tool_call_end_token in delta_text:
            logger.debug("tool_call_end_token in delta_text")
            full_text = current_text + delta_text
            tool_call_portion = full_text.split(
                self.tool_call_start_token)[-1].split(
                    self.tool_call_end_token)[0].rstrip()
            delta_text = delta_text.split(
                self.tool_call_end_token)[0].rstrip()
            text_portion = delta_text.split(
                self.tool_call_end_token)[-1].lstrip()

        # case -- we're starting a new tool call
        if (cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count):
            if len(delta_token_ids) > 1:
                tool_call_portion = current_text.split(
                    self.tool_call_start_token)[-1]
            else:
                tool_call_portion = None
                delta = None

            text_portion = None

            # set cursors and state appropriately
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            logger.debug("Starting on a new tool %s", self.current_tool_id)

        # case -- we're updating an existing tool call
        elif (cur_tool_start_count > cur_tool_end_count
              and cur_tool_start_count == prev_tool_start_count):

            # get the portion of the text that's the tool call
            tool_call_portion = current_text.split(
                self.tool_call_start_token)[-1]
            text_portion = None

        # case -- the current tool call is being closed.
        elif (cur_tool_start_count == cur_tool_end_count
              and cur_tool_end_count >= prev_tool_end_count):
            if self.prev_tool_call_arr is None or len(
                    self.prev_tool_call_arr) == 0:
                logger.debug(
                    "attempting to close tool call, but no tool call")
                return None
            
            # Handle final streaming of arguments if needed
            if self.current_tool_id < len(self.prev_tool_call_arr):
                prev_args = self.prev_tool_call_arr[self.current_tool_id].get("arguments", "")
                if prev_args and delta_text:
                    # Stream any remaining argument content
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=delta_text).model_dump(exclude_none=True),
                        )
                    ])

        # case -- otherwise we're just generating text
        else:
            text = delta_text.replace(self.tool_call_start_token, "")
            text = text.replace(self.tool_call_end_token, "")
            delta = DeltaMessage(tool_calls=[], content=text)
            return delta

        current_tool_call = dict()
        if tool_call_portion:
            current_tool_call_matches = (
                self.stream_tool_call_portion_regex.match(
                    tool_call_portion))
            if current_tool_call_matches:
                tool_name, tool_args = (
                    current_tool_call_matches.groups())
                current_tool_call["name"] = tool_name
                current_tool_call["arguments"] = tool_args
            else:
                current_tool_call_name_matches = (
                    self.stream_tool_call_name_regex.match(
                        tool_call_portion))
                if current_tool_call_name_matches:
                    tool_name = (
                        current_tool_call_name_matches.groups()[0])
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = ""
                else:
                    logger.debug("Not enough tokens for tool call parsing")
                    return None

        # case - we haven't sent the tool name yet. If it's available, send
        #   it. otherwise, wait until it's available.
        if not self.current_tool_name_sent:
            if current_tool_call is None:
                return None
            function_name: Union[str, None] = current_tool_call.get("name")
            if function_name:
                self.current_tool_name_sent = True
                
                # IMPORTANT: Add to prev_tool_call_arr immediately when we detect a tool call
                # This ensures finish_reason="tool_calls" even if parsing isn't complete
                already_added = any(
                    tool.get("name") == function_name
                    for tool in self.prev_tool_call_arr
                )
                if not already_added:
                    self.prev_tool_call_arr.append({
                        "name": function_name,
                        "arguments": "{}",  # Placeholder, will be updated later
                    })
                
                return DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=random_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=function_name).model_dump(
                                exclude_none=True),
                    )
                ])
            else:
                return None

        # case -- otherwise, send the tool call delta

        # if the tool call portion is None, send the delta as text
        if tool_call_portion is None:
            # if there's text but not tool calls, send that -
            # otherwise None to skip chunk
            delta = (DeltaMessage(
                content=delta_text) if text_portion is not None else None)
            return delta

        # now, the nitty-gritty of tool calls
        # now we have the portion to parse as tool call.

        logger.debug("Trying to parse current tool call with ID %s",
                     self.current_tool_id)

        # if we're starting a new tool call, push an empty object in as
        #   a placeholder for the arguments
        if len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})

        # main logic for tool parsing here - compare prev. partially-parsed
        #   arguments to the current partially-parsed arguments
        prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
            "arguments")
        cur_arguments = current_tool_call.get("arguments")

        logger.debug("diffing old arguments: %s", prev_arguments)
        logger.debug("against new ones: %s", cur_arguments)

        # case -- no arguments have been created yet. skip sending a delta.
        if not cur_arguments and not prev_arguments:
            logger.debug("Skipping text %s - no arguments", delta_text)
            delta = None

        # case -- prev arguments are defined, but none are now.
        #   probably impossible, but not a fatal error - just keep going
        elif not cur_arguments and prev_arguments:
            logger.error("should be impossible to have arguments reset "
                         "mid-call. skipping streaming anything.")
            delta = None

        # case -- we now have the first info about arguments available
        elif cur_arguments and not prev_arguments:
            # Update prev_tool_call_arr with the latest arguments
            if self.current_tool_id < len(self.prev_tool_call_arr):
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = cur_arguments

            delta = DeltaMessage(tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(
                        arguments=cur_arguments).model_dump(
                            exclude_none=True),
                )
            ])
            self.streamed_args_for_tool[
                self.current_tool_id] = cur_arguments

        # last case -- we have an update to existing arguments.
        elif cur_arguments and prev_arguments:
            if (isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)):
                delta_arguments = cur_arguments[len(prev_arguments):]
                logger.debug("got diff %s", delta_arguments)

                # Update prev_tool_call_arr with the latest arguments
                if self.current_tool_id < len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = cur_arguments

                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(
                            arguments=delta_arguments).model_dump(
                                exclude_none=True),
                    )
                ])
                self.streamed_args_for_tool[
                    self.current_tool_id] = cur_arguments
            else:
                delta = None

        # handle saving the state for the current tool into
        # the "prev" list for use in diffing for the next iteration
        if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
            self.prev_tool_call_arr[
                self.current_tool_id] = current_tool_call
        else:
            self.prev_tool_call_arr.append(current_tool_call)

        return delta

    def _handle_json_format_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> Union[DeltaMessage, None]:
        """Handle JSON format tool calls that appear as regular content."""
        
        # Add current delta to our buffer
        self.json_buffer += delta_text
        
        # Check if we're starting a JSON tool call
        if not self.in_json_tool_call:
            if '{"name":' in self.json_buffer or '{"name" :' in self.json_buffer:
                self.in_json_tool_call = True
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                
                # Find where the JSON starts and extract any content before it
                json_start_idx = self.json_buffer.find('{"name"')
                if json_start_idx == -1:
                    json_start_idx = self.json_buffer.find('{"name" ')
                
                if json_start_idx > 0:
                    # There's content before the JSON - return it as content
                    content_before = self.json_buffer[:json_start_idx]
                    self.json_buffer = self.json_buffer[json_start_idx:]
                    return DeltaMessage(content=content_before)
                else:
                    # No content before JSON, start processing the tool call
                    self.json_buffer = self.json_buffer[json_start_idx:] if json_start_idx >= 0 else self.json_buffer
        
        if not self.in_json_tool_call:
            # We're in regular content mode
            return DeltaMessage(content=delta_text)
            
        # We're in JSON tool call mode - try to parse what we have so far
        try:
            # Check if we have a complete JSON object
            if self.json_buffer.count('{') > 0 and self.json_buffer.count('{') == self.json_buffer.count('}'):
                # We might have a complete JSON - try to parse it
                match = self.json_tool_call_complete_regex.search(self.json_buffer)
                if match:
                    function_name = match.group('function_name')
                    function_arguments = match.group('function_arguments')
                    
                    # Parse the JSON arguments
                    try:
                        parsed_args = json.loads(function_arguments)
                        
                        # We have a complete tool call - reset state and return final chunk
                        self.in_json_tool_call = False
                        remaining_content = self.json_buffer[match.end():]
                        self.json_buffer = ""
                        
                        # First send the tool name if not sent yet
                        if not self.current_tool_name_sent:
                            self.current_tool_name_sent = True
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    type="function",
                                    id=random_tool_call_id(),
                                    function=DeltaFunctionCall(
                                        name=function_name).model_dump(exclude_none=True),
                                )
                            ])
                        else:
                            # Update prev_tool_call_arr with final arguments
                            if self.current_tool_id < len(self.prev_tool_call_arr):
                                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = json.dumps(parsed_args)
                            
                            # Send the complete arguments
                            delta_msg = DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=json.dumps(parsed_args)).model_dump(exclude_none=True),
                                )
                            ])
                            
                            # If there's remaining content, we'll handle it in the next call
                            if remaining_content.strip():
                                self.json_buffer = remaining_content
                                self.in_json_tool_call = False
                            
                            return delta_msg
                            
                    except json.JSONDecodeError:
                        # Invalid JSON - treat as regular content
                        self.in_json_tool_call = False
                        content = self.json_buffer
                        self.json_buffer = ""
                        return DeltaMessage(content=content)
                        
            # JSON is incomplete - check if we can extract the function name
            if not self.current_tool_name_sent:
                name_match = re.search(r'"name":\s*"([^"]+)"', self.json_buffer)
                if name_match:
                    function_name = name_match.group(1)
                    self.current_tool_name_sent = True
                    
                    # IMPORTANT: Add to prev_tool_call_arr immediately when we detect a tool call
                    # This ensures finish_reason="tool_calls" even if parsing isn't complete
                    already_added = any(
                        tool.get("name") == function_name
                        for tool in self.prev_tool_call_arr
                    )
                    if not already_added:
                        self.prev_tool_call_arr.append({
                            "name": function_name,
                            "arguments": "{}",  # Placeholder, will be updated later
                        })
                    
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=random_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=function_name).model_dump(exclude_none=True),
                        )
                    ])
                        
            # We don't have enough JSON yet - suppress output
            return None
            
        except Exception as e:
            logger.exception("Error parsing JSON tool call: %s", e)
            # Fall back to treating as regular content
            self.in_json_tool_call = False
            content = self.json_buffer
            self.json_buffer = ""
            return DeltaMessage(content=content)

    def _handle_xml_format_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> Union[DeltaMessage, None]:
        """Handle XML format tool calls with incremental streaming."""
        
        # Add current delta to our buffer
        self.xml_buffer += delta_text
        
        # Check if we're starting an XML tool call
        if not self.in_xml_tool_call:
            if '<function_calls>' in self.xml_buffer:
                self.in_xml_tool_call = True
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.xml_params = {}
                
                # Find where the XML starts and extract any content before it
                xml_start_idx = self.xml_buffer.find('<function_calls>')
                
                if xml_start_idx > 0:
                    # There's content before the XML - return it as content
                    content_before = self.xml_buffer[:xml_start_idx]
                    self.xml_buffer = self.xml_buffer[xml_start_idx:]
                    return DeltaMessage(content=content_before)
                else:
                    # No content before XML, start processing the tool call
                    self.xml_buffer = self.xml_buffer[xml_start_idx:] if xml_start_idx >= 0 else self.xml_buffer
        
        if not self.in_xml_tool_call:
            # We're in regular content mode
            return DeltaMessage(content=delta_text)
            
        # We're in XML tool call mode - parse incrementally
        try:
            # 1. First, try to extract function name if not sent yet
            if not self.current_tool_name_sent:
                invoke_match = re.search(r'<invoke\s+name="([^"]+)">', self.xml_buffer)
                if invoke_match:
                    self.xml_function_name = invoke_match.group(1)
                    self.current_tool_name_sent = True
                    
                    # IMPORTANT: Add to prev_tool_call_arr immediately when we detect a tool call
                    # This ensures finish_reason="tool_calls" even if parsing isn't complete
                    already_added = any(
                        tool.get("name") == self.xml_function_name
                        for tool in self.prev_tool_call_arr
                    )
                    if not already_added:
                        self.prev_tool_call_arr.append({
                            "name": self.xml_function_name,
                            "arguments": "{}",  # Placeholder, will be updated later
                        })
                    
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=random_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=self.xml_function_name).model_dump(exclude_none=True),
                        )
                    ])
                else:
                    # Function name not ready yet, suppress output
                    return None
            
            # 2. Look for complete parameters and stream them incrementally
            # Find all complete parameters in the current buffer
            param_matches = self.xml_parameter_regex.findall(self.xml_buffer)
            current_found_params = {}
            for param_name, param_value in param_matches:
                current_found_params[param_name] = param_value.strip()
            
            # Check if we have new parameters compared to what we already have
            if current_found_params != self.xml_params:
                # Update our accumulated parameters
                self.xml_params = current_found_params.copy()
                
                # Stream the updated parameters as JSON arguments
                current_args_json = json.dumps(self.xml_params)
                
                # Get the previous arguments that were streamed
                prev_args_json = self.streamed_args_for_tool[self.current_tool_id] if self.current_tool_id < len(self.streamed_args_for_tool) else ""
                
                # Only stream if the JSON has actually changed
                if current_args_json != prev_args_json:
                    # Calculate the delta to stream (just the new part)
                    if prev_args_json == "":
                        # First time streaming arguments - send the complete object
                        delta_args = current_args_json
                    else:
                        # Calculate the textual difference for streaming
                        # For OpenAI compatibility, we send just the new characters
                        if current_args_json.startswith(prev_args_json):
                            delta_args = current_args_json[len(prev_args_json):]
                        else:
                            # JSON structure changed (shouldn't happen normally), send complete new JSON
                            delta_args = current_args_json
                    
                    # Update what we've streamed so far
                    self.streamed_args_for_tool[self.current_tool_id] = current_args_json
                    
                    # Only send if there's actually a delta
                    if delta_args:
                        # Update prev_tool_call_arr with the latest arguments
                        if self.current_tool_id < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = current_args_json
                        
                        return DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=delta_args).model_dump(exclude_none=True),
                            )
                        ])
            
            # 3. Check if we're done with the tool call
            if '</function_calls>' in self.xml_buffer:
                # Tool call is complete, reset state
                self.in_xml_tool_call = False
                remaining_content_match = re.search(r'</function_calls>(.*)', self.xml_buffer, re.DOTALL)
                if remaining_content_match:
                    remaining_content = remaining_content_match.group(1)
                    if remaining_content.strip():
                        # There's content after the tool call, reset buffer for next processing
                        self.xml_buffer = remaining_content
                        # Don't return anything here, let the next call handle remaining content
                        return None
                else:
                    self.xml_buffer = ""
                
                # Tool call finished, no more deltas to send
                return None
            
            # 4. Still building the XML, suppress output for now
            return None
            
        except Exception as e:
            logger.exception("Error parsing XML tool call: %s", e)
            # Fall back to treating as regular content
            self.in_xml_tool_call = False
            content = self.xml_buffer
            self.xml_buffer = ""
            return DeltaMessage(content=content)
