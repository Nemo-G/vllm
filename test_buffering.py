#!/usr/bin/env python3

# Simple test for XML buffering logic
class MockParser:
    def __init__(self):
        self.parameter_end_token = "</parameter>"
        self.function_end_token = "</function>"
        self.tool_call_end_token = "</tool_call>"
        self.xml_buffer = ""
        self.buffering_xml = False

    def _process_xml_buffer(self, delta_text: str) -> tuple[str, bool]:
        """
        Process XML buffering to handle partial end tags and newlines.
        
        Key principle: ALWAYS split at newlines and buffer them, then check if
        they're followed by end tags to determine if they should be stripped.
        
        Returns:
            tuple[str, bool]: (processed_delta_text, should_continue_processing)
            - processed_delta_text: The delta text after buffering processing
            - should_continue_processing: Whether to continue processing
        """
        # Known end tokens we need to buffer for
        end_tokens = [self.parameter_end_token, self.function_end_token, self.tool_call_end_token]
        
        # If we're not buffering, check for complete end tokens first
        if not self.buffering_xml:
            for token in end_tokens:
                if delta_text == token:
                    # Complete end token, pass through immediately
                    return delta_text, True
        
        # If we're not buffering and delta has no special characters, pass through
        if not self.buffering_xml and '<' not in delta_text and '\n' not in delta_text:
            return delta_text, True
        
        output_content = ""
        
        # If we're already buffering, process the new delta
        if self.buffering_xml:
            # Add new delta to buffer
            self.xml_buffer += delta_text
            
            # Check if we now have a complete end token
            for token in end_tokens:
                if self.xml_buffer == token:
                    # Complete end token, output it
                    self.buffering_xml = False
                    result = self.xml_buffer
                    self.xml_buffer = ""
                    return result, True
            
            # Check if we have newline + complete end token (preserve the newline)
            for token in end_tokens:
                if self.xml_buffer == '\n' + token:
                    # Newline + end token, output both
                    self.buffering_xml = False
                    result = self.xml_buffer
                    self.xml_buffer = ""
                    return result, True
            
            # Quick fail: if buffer is just '<' and new input doesn't start with '/'
            if self.xml_buffer == '<' and not delta_text.startswith('/'):
                # Not an end tag, output buffered '<' and process delta normally
                self.buffering_xml = False
                buffered_content = self.xml_buffer
                self.xml_buffer = ""
                
                # Process the delta_text normally (it might contain more special chars)
                delta_result, delta_continue = self._process_xml_buffer(delta_text)
                return buffered_content + delta_result, delta_continue
            
            # Check if buffer could still become valid
            could_be_valid = False
            for token in end_tokens:
                if token.startswith(self.xml_buffer) or ('\n' + token).startswith(self.xml_buffer):
                    could_be_valid = True
                    break
            
            # Quick fail: if buffer starts with newline and next part doesn't look like end tag
            if (self.xml_buffer.startswith('\n') and len(self.xml_buffer) > 1 and 
                not self.xml_buffer[1:].startswith('<')):
                could_be_valid = False
                
                # Quick fail detected - need to handle additional newlines in the input
                self.buffering_xml = False
                
                # Find where the original buffer ended and new delta started
                original_buffer_len = len(self.xml_buffer) - len(delta_text)
                buffered_part = self.xml_buffer[:original_buffer_len]  # Just the original "\n"
                delta_part = self.xml_buffer[original_buffer_len:]     # The new delta_text
                
                self.xml_buffer = ""
                
                # Check if delta_part contains newlines that need new buffering
                if '\n' in delta_part:
                    newline_pos = delta_part.find('\n')
                    content_before_newline = delta_part[:newline_pos]
                    content_from_newline = delta_part[newline_pos:]
                    
                    # Output buffered newline + content before new newline
                    output_result = buffered_part + content_before_newline
                    
                    # Start new buffering from the newline
                    self.buffering_xml = True
                    self.xml_buffer = content_from_newline
                    
                    return output_result, len(output_result) > 0
                else:
                    # No additional newlines, output everything
                    return buffered_part + delta_part, True
            
            max_reasonable_len = max(len(token) for token in end_tokens) + 2  # +2 for newline + some margin
            if could_be_valid and len(self.xml_buffer) <= max_reasonable_len:
                # Keep buffering
                return "", False
            else:
                # Buffer is invalid or too long
                self.buffering_xml = False
                
                # Check if this is due to invalid XML structure (not newline-related)
                if not could_be_valid and not self.xml_buffer.startswith('\n'):
                    # Extract original buffer and new delta
                    original_buffer_len = len(self.xml_buffer) - len(delta_text)
                    original_buffer = self.xml_buffer[:original_buffer_len]
                    
                    # Check if new delta should start new buffering
                    if '<' in delta_text or '\n' in delta_text:
                        # Start new buffering with the delta
                        self.buffering_xml = True
                        self.xml_buffer = delta_text
                        return original_buffer, len(original_buffer) > 0
                    else:
                        # No special chars in delta, output everything
                        result = self.xml_buffer
                        self.xml_buffer = ""
                        return result, True
                else:
                    # Output entire buffer for other cases (too long, etc.)
                    result = self.xml_buffer
                    self.xml_buffer = ""
                    return result, True
        
        # Not currently buffering - check if we need to start
        if '\n' in delta_text:
            # ALWAYS split at newlines - this is the key fix
            newline_pos = delta_text.find('\n')
            output_content = delta_text[:newline_pos]  # Everything before first newline
            
            # Start buffering from the newline onwards
            self.buffering_xml = True
            self.xml_buffer = delta_text[newline_pos:]
            
            # Return output and indicate we should NOT continue processing 
            # (because we're now buffering)
            return output_content, len(output_content) > 0
        
        elif '<' in delta_text:
            # Found start of XML tag
            lt_pos = delta_text.find('<')
            output_content = delta_text[:lt_pos]
            
            # Check if this could be a valid end tag start
            remaining = delta_text[lt_pos:]
            if len(remaining) == 1:  # Just '<' - could be followed by '/' in next chunk
                # Start buffering - we'll quickly fail if next chunk doesn't start with '/'
                self.buffering_xml = True
                self.xml_buffer = remaining
                return output_content, len(output_content) > 0
            else:
                # More content after '<', check if it looks like an end tag
                if remaining.startswith('</'):
                    # This is an end tag, start buffering
                    self.buffering_xml = True
                    self.xml_buffer = remaining
                    return output_content, len(output_content) > 0
                else:
                    # Not an end tag, pass through
                    return delta_text, True
        
        else:
            # No special characters, pass through
            return delta_text, True

def test_sequential_calls(parser, inputs, description):
    """Helper function to test sequential calls to the parser"""
    print(f"\n{description}")
    outputs = []
    continues = []
    
    for i, inp in enumerate(inputs):
        result, continue_processing = parser._process_xml_buffer(inp)
        outputs.append(result)
        continues.append(continue_processing)
        print(f"  Step {i+1}: Input: '{inp}' -> Output: '{result}', Continue: {continue_processing}")
    
    return outputs, continues

def test_buffering():
    """
    Test XML buffering logic for handling partial end tags in streaming.
    
    Note: In practice, when models output tool calls, literal '<' characters 
    in parameter values should be encoded as &lt; so raw '<' characters are 
    likely to be XML tags. This buffering handles the case where XML end tags 
    like </parameter> are split across multiple streaming chunks.
    """
    parser = MockParser()
    
    # Test case 1: Normal text (no buffering needed)
    print("Test 1: Normal text")
    result, continue_processing = parser._process_xml_buffer("hello world")
    print(f"  Input: 'hello world' -> Output: '{result}', Continue: {continue_processing}")
    assert result == "hello world" and continue_processing
    
    # Test case 2: Complete end token in one chunk
    print("\nTest 2: Complete end token")
    result, continue_processing = parser._process_xml_buffer("</parameter>")
    print(f"  Input: '</parameter>' -> Output: '{result}', Continue: {continue_processing}")
    assert result == "</parameter>" and continue_processing
    
    # Test case 3: Partial end token (should buffer)
    print("\nTest 3: Partial end token")
    result, continue_processing = parser._process_xml_buffer("</")
    print(f"  Input: '</' -> Output: '{result}', Continue: {continue_processing}")
    assert result == "" and not continue_processing
    
    # Test case 4: Complete the buffered token
    print("Test 4: Complete buffered token")
    result, continue_processing = parser._process_xml_buffer("parameter>")
    print(f"  Input: 'parameter>' -> Output: '{result}', Continue: {continue_processing}")
    assert result == "</parameter>" and continue_processing
    
    # Test case 5: Mixed content with partial tag
    parser = MockParser()  # Reset
    print("\nTest 5: Mixed content with partial tag")
    result, continue_processing = parser._process_xml_buffer("hello</")
    print(f"  Input: 'hello</' -> Output: '{result}', Continue: {continue_processing}")
    assert result == "hello" and continue_processing
    
    # Test case 6: Complete the mixed content tag
    print("Test 6: Complete mixed content tag")
    result, continue_processing = parser._process_xml_buffer("parameter>")
    print(f"  Input: 'parameter>' -> Output: '{result}', Continue: {continue_processing}")
    assert result == "</parameter>" and continue_processing
    
    # Test case 7: Invalid partial tag (should output after timeout)
    parser = MockParser()  # Reset  
    print("\nTest 7: Invalid partial tag")
    result, continue_processing = parser._process_xml_buffer("<invalid_very_long_tag")
    print(f"  Input: '<invalid_very_long_tag' -> Output: '{result}', Continue: {continue_processing}")
    # Should buffer '<', then quick fail when seeing 'invalid...', outputting everything
    assert result == "<invalid_very_long_tag" and continue_processing
    
    # Second input should also pass through normally
    result2, continue_processing2 = parser._process_xml_buffer("_more_invalid_content")
    print(f"  Input: '_more_invalid_content' -> Output: '{result2}', Continue: {continue_processing2}")
    assert result2 == "_more_invalid_content" and continue_processing2
    
    # Test case 8: Sequential partial inputs building complete tag
    parser = MockParser()  # Reset
    inputs = ["<", "/", "param", "eter", ">"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 8: Sequential partial inputs")
    # Should buffer until complete, then output the complete tag
    expected_outputs = ["", "", "", "", "</parameter>"]
    expected_continues = [False, False, False, False, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 9: Multiple '<' characters then valid end tag
    parser = MockParser()  # Reset
    inputs = ["<", "<", "</", "param", "eter", ">"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 9: Multiple '<' characters then valid tag")
    # First '<' starts buffering, second '<' triggers quick fail, then '</' starts new buffering
    expected_outputs = ["", "<", "<", "", "", "</parameter>"]
    expected_continues = [False, True, True, False, False, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 10: Mixed content during buffering
    parser = MockParser()  # Reset
    inputs = ["content</", "par", "ameter>more"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 10: Mixed content during buffering")
    # Should output "content", then buffer, then complete tag + "more"
    assert outputs[0] == "content"
    assert outputs[2] == "</parameter>more"
    assert continues == [True, False, True]
    
    # Test case 11: False start then real tag
    parser = MockParser()  # Reset
    inputs = ["<false", "start", "<", "/", "parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 11: False start then real tag")
    # "<false" and "start" pass through, then "<" "/" "parameter>" form valid end tag
    expected_outputs = ["<false", "start", "", "", "</parameter>"]
    expected_continues = [True, True, False, False, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 12: Continuous streaming of complete tag
    parser = MockParser()  # Reset
    inputs = ["<", "/", "f", "u", "n", "c", "t", "i", "o", "n", ">"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 12: Streaming complete </function> tag")
    # All should buffer except the last which outputs complete tag
    expected_continues = [False] * 10 + [True]
    expected_final_output = "</function>"
    assert continues == expected_continues
    assert outputs[-1] == expected_final_output
    
    # Test case 13: Wrong tag that becomes too long
    parser = MockParser()  # Reset
    inputs = ["<", "/", "very", "long", "invalid", "tag", "name"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 13: Invalid long tag")
    # Buffers "</" and "very" then gives up when too long, outputting "</very" then rest normally
    expected_outputs = ["", "", "</very", "long", "invalid", "tag", "name"]
    expected_continues = [False, False, True, True, True, True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 14: Nested content with multiple potential tags
    parser = MockParser()  # Reset
    inputs = ["text<", "/par", "tial<", "/parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 14: Nested partial tags")
    # All inputs pass through since none start with '</'
    print(outputs,continues)
    expected_outputs = ["text", "", "</par", "tial</parameter>"]
    expected_continues = [True, False, True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # NEWLINE HANDLING TESTS
    
    # Test case 15: Content + newline followed by end tag
    parser = MockParser()  # Reset
    inputs = ["San Francisco\n", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 15: Content + newline before end tag")
    # Should buffer the newline, then when it sees end tag, preserve the newline
    expected_outputs = ["San Francisco", "\n</parameter>"]
    expected_continues = [True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 16: Just newline followed by end tag
    parser = MockParser()  # Reset
    inputs = ["content\n\n", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 16: Just newline before end tag")
    # Should buffer the newline, then combine with end tag preserving the newline
    expected_outputs = ["content", "\n\n</parameter>"]
    expected_continues = [True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 17: Newline not followed by end tag (quick fail path)
    parser = MockParser()  # Reset
    inputs = ["content\n", "more content"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 17: Newline not followed by end tag (quick fail)")
    # Should buffer newline initially, then quick fail and output it when next chunk doesn't start with '<'
    expected_outputs = ["content", "\nmore content"]
    expected_continues = [True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 18: Multiple newlines and content
    parser = MockParser()  # Reset
    inputs = ["line1\n", "\n", "line2\n", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 18: Multiple newlines with end tag")
    # Should handle multiple newlines correctly:
    # 1. "line1\n" → "line1" (split at newline)
    # 2. "\n" → "\n" (quick fail: buffered "\n" + "\n" doesn't start with '<')
    # 3. "line2\n" → "\nline2" (quick fail: buffered "\n" + "line2\n" doesn't start with '<')
    # 4. "</parameter>" → "</parameter>" (strip formatting newline before end tag)
    expected_outputs = ["line1", "\n", "\nline2", "\n</parameter>"]
    expected_continues = [True, True, True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 19: Content newline vs formatting newline
    parser = MockParser()  # Reset
    inputs = ["Address:\n123 Main St\n", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 19: Content newlines vs formatting")
    # Should preserve content newlines:
    # 1. "Address:\n123 Main St\n" → "Address:" (split at first newline)
    # 2. "</parameter>" → "\n123 Main St\n</parameter>" (buffer has content between newline and end tag, so quick fail outputs all)
    expected_outputs = ["Address:", "\n123 Main St\n</parameter>"]
    expected_continues = [True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 20: Quick fail path with just newline
    parser = MockParser()  # Reset
    inputs = ["\n", "regular content"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 20: Quick fail path - newline + non-tag content")
    # Should buffer newline, then immediately output it when next chunk doesn't start with '<'
    expected_outputs = ["", "\nregular content"]
    expected_continues = [False, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 21: Quick fail vs successful end tag detection
    parser = MockParser()  # Reset
    inputs = ["\n", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 21: Newline followed by end tag (preserve newline)")
    # Should buffer newline, then when it sees end tag, preserve the newline and output newline + tag
    expected_outputs = ["", "\n</parameter>"]
    expected_continues = [False, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 22: Mid-string newline detection and buffering
    parser = MockParser()  # Reset
    inputs = ["San Francisco\nCA", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 22: Mid-string newline detection")
    # Should split at the newline: output "San Francisco", buffer "\nCA"
    # Then when it sees end tag, output buffered content + end tag
    expected_outputs = ["San Francisco", "\nCA</parameter>"]
    expected_continues = [True, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    # Test case 23: User's exact scenario - content with trailing newline then end tag
    parser = MockParser()  # Reset
    inputs = ["\nvalue\n", "</parameter>"]
    outputs, continues = test_sequential_calls(parser, inputs, "Test 23: User scenario - value + newline + end tag")
    expected_outputs = ["", "\nvalue\n</parameter>"]
    expected_continues = [False, True]
    assert outputs == expected_outputs and continues == expected_continues
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    # Test cases for specific edge cases
    
    # Test A: Quick fail with additional newlines in delta_text  
    print("Test A: Quick fail + additional newlines in delta_text")
    parser = MockParser()
    
    # Step 1: Buffer a newline
    result1, continue1 = parser._process_xml_buffer("content\n")
    print(f"Step 1: Input 'content\\n' → Output: '{result1}', Buffer: '{parser.xml_buffer}', Continue: {continue1}")
    
    # Step 2: Quick fail with additional newline in delta_text
    result2, continue2 = parser._process_xml_buffer("more content\nanother line")
    print(f"Step 2: Input 'more content\\nanother line' → Output: '{result2}', Buffer: '{parser.xml_buffer}', Continue: {continue2}")
    
    # Expected: 
    # - result1 should be "content" 
    # - result2 should be "\nmore content" (buffered newline + content before new newline)
    # - buffer should be "\nanother line" (new buffering from the newline)
    
    expected_result1 = "content"
    expected_result2 = "\nmore content" 
    expected_buffer = "\nanother line"
    
    success = (result1 == expected_result1 and 
               result2 == expected_result2 and 
               parser.xml_buffer == expected_buffer)
    
    print(f"Expected: result1='{expected_result1}', result2='{expected_result2}', buffer='{expected_buffer}'")
    print(f"Got:      result1='{result1}', result2='{result2}', buffer='{parser.xml_buffer}'")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    
    # Test B: User's exact scenario - content with trailing newline followed by end tag
    print("\nTest B: User's exact scenario")
    parser = MockParser()
    
    # Step 1: Process "San Francisco, CA\n"
    result1, continue1 = parser._process_xml_buffer("San Francisco, CA\n")
    print(f"Step 1: Input 'San Francisco, CA\\n' → Output: '{result1}', Buffer: '{parser.xml_buffer}', Continue: {continue1}")
    
    # Step 2: Process "</parameter>"
    result2, continue2 = parser._process_xml_buffer("</parameter>")
    print(f"Step 2: Input '</parameter>' → Output: '{result2}', Buffer: '{parser.xml_buffer}', Continue: {continue2}")
    
    # Verify results
    expected_result1 = "San Francisco, CA"  # Content before newline
    expected_result2 = "</parameter>"       # End tag without newline
    
    success = (result1 == expected_result1 and result2 == expected_result2)
    print(f"Expected: result1='{expected_result1}', result2='{expected_result2}'")
    print(f"Got:      result1='{result1}', result2='{result2}'")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    
    # Test C: Multiple quick fails
    print("\nTest C: Multiple quick fails")
    parser = MockParser()
    
    results = []
    inputs = ["line1\n", "content\n", "more\n", "text"]
    
    for i, inp in enumerate(inputs):
        result, cont = parser._process_xml_buffer(inp)
        results.append(result)
        print(f"Step {i+1}: Input '{inp}' → Output: '{result}', Buffer: '{parser.xml_buffer}', Continue: {cont}")
    
    # Expected: each newline should trigger quick fail and be preserved as content
    expected = ["line1", "\ncontent", "\nmore", "\ntext"]
    success = results == expected
    print(f"Expected: {expected}")
    print(f"Got:      {results}")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")

    print("\n" + "="*50)
    test_buffering() 