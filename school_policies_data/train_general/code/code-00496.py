        # TODO: support for BOM within a stream.
        #if ch == '\uFEFF':
        #    return self.fetch_bom()    <-- issue BOMToken

        # Note: the order of the following checks is NOT significant.

        # Is it the flow sequence start indicator?
        if ch == '[':
            return self.fetch_flow_sequence_start()

        # Is it the flow mapping start indicator?
        if ch == '{':
            return self.fetch_flow_mapping_start()

        # Is it the flow sequence end indicator?
        if ch == ']':
            return self.fetch_flow_sequence_end()

        # Is it the flow mapping end indicator?
        if ch == '}':
            return self.fetch_flow_mapping_end()

        # Is it the flow entry indicator?
        if ch == ',':
            return self.fetch_flow_entry()

        # Is it the block entry indicator?
        if ch == '-' and self.check_block_entry():
            return self.fetch_block_entry()

        # Is it the key indicator?
        if ch == '?' and self.check_key():
            return self.fetch_key()

        # Is it the value indicator?
        if ch == ':' and self.check_value():
            return self.fetch_value()

        # Is it an alias?
        if ch == '*':
            return self.fetch_alias()

        # Is it an anchor?
        if ch == '&':
            return self.fetch_anchor()

        # Is it a tag?
        if ch == '!':
            return self.fetch_tag()

        # Is it a literal scalar?
        if ch == '|' and not self.flow_level:
            return self.fetch_literal()

        # Is it a folded scalar?
        if ch == '>' and not self.flow_level:
            return self.fetch_folded()

        # Is it a single quoted scalar?
        if ch == '\'':
            return self.fetch_single()

        # Is it a double quoted scalar?
        if ch == '\"':
            return self.fetch_double()

        # It must be a plain scalar then.
        if self.check_plain():
            return self.fetch_plain()

        # No? It's an error. Let's produce a nice error message.
        raise ScannerError("while scanning for the next token", None,
                "found character %r that cannot start any token" % ch,
                self.get_mark())

    # Simple keys treatment.

    def next_possible_simple_key(self):
        # Return the number of the nearest possible simple key. Actually we
        # don't need to loop through the whole dictionary. We may replace it
        # with the following code:
        #   if not self.possible_simple_keys:
        #       return None
        #   return self.possible_simple_keys[
        #           min(self.possible_simple_keys.keys())].token_number
        min_token_number = None
        for level in self.possible_simple_keys:
            key = self.possible_simple_keys[level]
            if min_token_number is None or key.token_number < min_token_number:
                min_token_number = key.token_number
        return min_token_number

    def stale_possible_simple_keys(self):
        # Remove entries that are no longer possible simple keys. According to
        # the YAML specification, simple keys
        # - should be limited to a single line,
        # - should be no longer than 1024 characters.
        # Disabling this procedure will allow simple keys of any length and
        # height (may cause problems if indentation is broken though).
        for level in list(self.possible_simple_keys):
            key = self.possible_simple_keys[level]
            if key.line != self.line  \
                    or self.index-key.index > 1024:
                if key.required:
                    raise ScannerError("while scanning a simple key", key.mark,
                            "could not find expected ':'", self.get_mark())
                del self.possible_simple_keys[level]

    def save_possible_simple_key(self):
        # The next token may start a simple key. We check if it's possible
        # and save its position. This function is called for
        #   ALIAS, ANCHOR, TAG, SCALAR(flow), '[', and '{'.

        # Check if a simple key is required at the current position.
        required = not self.flow_level and self.indent == self.column

        # The next token might be a simple key. Let's save it's number and
        # position.
        if self.allow_simple_key:
            self.remove_possible_simple_key()
            token_number = self.tokens_taken+len(self.tokens)
            key = SimpleKey(token_number, required,
                    self.index, self.line, self.column, self.get_mark())
            self.possible_simple_keys[self.flow_level] = key

    def remove_possible_simple_key(self):
        # Remove the saved possible key position at the current flow level.
        if self.flow_level in self.possible_simple_keys:
            key = self.possible_simple_keys[self.flow_level]
            
            if key.required:
                raise ScannerError("while scanning a simple key", key.mark,
                        "could not find expected ':'", self.get_mark())

            del self.possible_simple_keys[self.flow_level]

    # Indentation functions.

    def unwind_indent(self, column):

        ## In flow context, tokens should respect indentation.
        ## Actually the condition should be `self.indent >= column` according to
        ## the spec. But this condition will prohibit intuitively correct
        ## constructions such as
        ## key : {
        ## }
        #if self.flow_level and self.indent > column:
        #    raise ScannerError(None, None,
        #            "invalid indentation or unclosed '[' or '{'",
        #            self.get_mark())

        # In the flow context, indentation is ignored. We make the scanner less
        # restrictive then specification requires.
        if self.flow_level:
            return

        # In block context, we may need to issue the BLOCK-END tokens.
        while self.indent > column:
            mark = self.get_mark()
            self.indent = self.indents.pop()
            self.tokens.append(BlockEndToken(mark, mark))

    def add_indent(self, column):
        # Check if we need to increase indentation.
        if self.indent < column:
            self.indents.append(self.indent)
            self.indent = column
            return True
        return False

    # Fetchers.

    def fetch_stream_start(self):
        # We always add STREAM-START as the first token and STREAM-END as the
        # last token.

        # Read the token.
        mark = self.get_mark()
        
        # Add STREAM-START.
        self.tokens.append(StreamStartToken(mark, mark,
            encoding=self.encoding))
        

    def fetch_stream_end(self):

        # Set the current indentation to -1.
        self.unwind_indent(-1)

        # Reset simple keys.
        self.remove_possible_simple_key()
        self.allow_simple_key = False
        self.possible_simple_keys = {}

        # Read the token.
        mark = self.get_mark()
        
        # Add STREAM-END.
        self.tokens.append(StreamEndToken(mark, mark))

        # The steam is finished.
        self.done = True

    def fetch_directive(self):
        
        # Set the current indentation to -1.
        self.unwind_indent(-1)

        # Reset simple keys.
        self.remove_possible_simple_key()
        self.allow_simple_key = False

        # Scan and add DIRECTIVE.
        self.tokens.append(self.scan_directive())

    def fetch_document_start(self):
        self.fetch_document_indicator(DocumentStartToken)

    def fetch_document_end(self):
        self.fetch_document_indicator(DocumentEndToken)

    def fetch_document_indicator(self, TokenClass):

        # Set the current indentation to -1.
        self.unwind_indent(-1)

        # Reset simple keys. Note that there could not be a block collection
        # after '---'.
        self.remove_possible_simple_key()
        self.allow_simple_key = False

        # Add DOCUMENT-START or DOCUMENT-END.
        start_mark = self.get_mark()
        self.forward(3)
        end_mark = self.get_mark()
        self.tokens.append(TokenClass(start_mark, end_mark))

    def fetch_flow_sequence_start(self):
        self.fetch_flow_collection_start(FlowSequenceStartToken)

    def fetch_flow_mapping_start(self):
        self.fetch_flow_collection_start(FlowMappingStartToken)

    def fetch_flow_collection_start(self, TokenClass):

        # '[' and '{' may start a simple key.
        self.save_possible_simple_key()

        # Increase the flow level.
        self.flow_level += 1

        # Simple keys are allowed after '[' and '{'.
        self.allow_simple_key = True

        # Add FLOW-SEQUENCE-START or FLOW-MAPPING-START.
        start_mark = self.get_mark()
        self.forward()
        end_mark = self.get_mark()
        self.tokens.append(TokenClass(start_mark, end_mark))

    def fetch_flow_sequence_end(self):
        self.fetch_flow_collection_end(FlowSequenceEndToken)

    def fetch_flow_mapping_end(self):
        self.fetch_flow_collection_end(FlowMappingEndToken)

    def fetch_flow_collection_end(self, TokenClass):

        # Reset possible simple key on the current level.
        self.remove_possible_simple_key()

        # Decrease the flow level.
        self.flow_level -= 1

        # No simple keys after ']' or '}'.
        self.allow_simple_key = False

        # Add FLOW-SEQUENCE-END or FLOW-MAPPING-END.
        start_mark = self.get_mark()
        self.forward()
        end_mark = self.get_mark()
        self.tokens.append(TokenClass(start_mark, end_mark))

    def fetch_flow_entry(self):

        # Simple keys are allowed after ','.
        self.allow_simple_key = True

        # Reset possible simple key on the current level.
        self.remove_possible_simple_key()

        # Add FLOW-ENTRY.
        start_mark = self.get_mark()
        self.forward()
        end_mark = self.get_mark()
        self.tokens.append(FlowEntryToken(start_mark, end_mark))

    def fetch_block_entry(self):

        # Block context needs additional checks.
        if not self.flow_level:

            # Are we allowed to start a new entry?
            if not self.allow_simple_key:
                raise ScannerError(None, None,
                        "sequence entries are not allowed here",
                        self.get_mark())

            # We may need to add BLOCK-SEQUENCE-START.
            if self.add_indent(self.column):
                mark = self.get_mark()
                self.tokens.append(BlockSequenceStartToken(mark, mark))

        # It's an error for the block entry to occur in the flow context,
        # but we let the parser detect this.
        else:
            pass

        # Simple keys are allowed after '-'.
        self.allow_simple_key = True

        # Reset possible simple key on the current level.
        self.remove_possible_simple_key()

        # Add BLOCK-ENTRY.
        start_mark = self.get_mark()
        self.forward()
        end_mark = self.get_mark()
        self.tokens.append(BlockEntryToken(start_mark, end_mark))

    def fetch_key(self):
        
        # Block context needs additional checks.
        if not self.flow_level:

            # Are we allowed to start a key (not necessary a simple)?
            if not self.allow_simple_key:
                raise ScannerError(None, None,
                        "mapping keys are not allowed here",
                        self.get_mark())

            # We may need to add BLOCK-MAPPING-START.
            if self.add_indent(self.column):
                mark = self.get_mark()
                self.tokens.append(BlockMappingStartToken(mark, mark))

        # Simple keys are allowed after '?' in the block context.
        self.allow_simple_key = not self.flow_level

        # Reset possible simple key on the current level.
        self.remove_possible_simple_key()

        # Add KEY.
        start_mark = self.get_mark()
        self.forward()
        end_mark = self.get_mark()
        self.tokens.append(KeyToken(start_mark, end_mark))

    def fetch_value(self):

        # Do we determine a simple key?
        if self.flow_level in self.possible_simple_keys:

            # Add KEY.
            key = self.possible_simple_keys[self.flow_level]
            del self.possible_simple_keys[self.flow_level]
            self.tokens.insert(key.token_number-self.tokens_taken,
                    KeyToken(key.mark, key.mark))

            # If this key starts a new block mapping, we need to add
            # BLOCK-MAPPING-START.
            if not self.flow_level:
                if self.add_indent(key.column):
                    self.tokens.insert(key.token_number-self.tokens_taken,
                            BlockMappingStartToken(key.mark, key.mark))

            # There cannot be two simple keys one after another.
            self.allow_simple_key = False

        # It must be a part of a complex key.
        else:
            
            # Block context needs additional checks.
            # (Do we really need them? They will be caught by the parser
            # anyway.)
            if not self.flow_level:

                # We are allowed to start a complex value if and only if
                # we can start a simple key.
                if not self.allow_simple_key:
                    raise ScannerError(None, None,
                            "mapping values are not allowed here",
                            self.get_mark())

            # If this value starts a new block mapping, we need to add
            # BLOCK-MAPPING-START.  It will be detected as an error later by
            # the parser.
            if not self.flow_level:
                if self.add_indent(self.column):
                    mark = self.get_mark()
                    self.tokens.append(BlockMappingStartToken(mark, mark))

            # Simple keys are allowed after ':' in the block context.
            self.allow_simple_key = not self.flow_level

            # Reset possible simple key on the current level.
            self.remove_possible_simple_key()

        # Add VALUE.
        start_mark = self.get_mark()
        self.forward()
        end_mark = self.get_mark()
        self.tokens.append(ValueToken(start_mark, end_mark))

    def fetch_alias(self):

        # ALIAS could be a simple key.
        self.save_possible_simple_key()

        # No simple keys after ALIAS.
        self.allow_simple_key = False

        # Scan and add ALIAS.
        self.tokens.append(self.scan_anchor(AliasToken))

    def fetch_anchor(self):

        # ANCHOR could start a simple key.
        self.save_possible_simple_key()

        # No simple keys after ANCHOR.
        self.allow_simple_key = False

        # Scan and add ANCHOR.
        self.tokens.append(self.scan_anchor(AnchorToken))

    def fetch_tag(self):

        # TAG could start a simple key.
        self.save_possible_simple_key()

        # No simple keys after TAG.
        self.allow_simple_key = False

        # Scan and add TAG.
        self.tokens.append(self.scan_tag())

    def fetch_literal(self):
        self.fetch_block_scalar(style='|')

    def fetch_folded(self):
        self.fetch_block_scalar(style='>')

    def fetch_block_scalar(self, style):

        # A simple key may follow a block scalar.
        self.allow_simple_key = True

        # Reset possible simple key on the current level.
        self.remove_possible_simple_key()

        # Scan and add SCALAR.
        self.tokens.append(self.scan_block_scalar(style))

    def fetch_single(self):
        self.fetch_flow_scalar(style='\'')

    def fetch_double(self):
        self.fetch_flow_scalar(style='"')

    def fetch_flow_scalar(self, style):

        # A flow scalar could be a simple key.
        self.save_possible_simple_key()

        # No simple keys after flow scalars.
        self.allow_simple_key = False

        # Scan and add SCALAR.
        self.tokens.append(self.scan_flow_scalar(style))

    def fetch_plain(self):

        # A plain scalar could be a simple key.
        self.save_possible_simple_key()

        # No simple keys after plain scalars. But note that `scan_plain` will
        # change this flag if the scan is finished at the beginning of the
        # line.
        self.allow_simple_key = False

        # Scan and add SCALAR. May change `allow_simple_key`.
        self.tokens.append(self.scan_plain())

    # Checkers.

    def check_directive(self):

        # DIRECTIVE:        ^ '%' ...
        # The '%' indicator is already checked.
        if self.column == 0:
            return True

    def check_document_start(self):

        # DOCUMENT-START:   ^ '---' (' '|'\n')
        if self.column == 0:
            if self.prefix(3) == '---'  \
                    and self.peek(3) in '\0 \t\r\n\x85\u2028\u2029':
                return True

    def check_document_end(self):

        # DOCUMENT-END:     ^ '...' (' '|'\n')
        if self.column == 0:
            if self.prefix(3) == '...'  \
                    and self.peek(3) in '\0 \t\r\n\x85\u2028\u2029':
                return True

    def check_block_entry(self):

        # BLOCK-ENTRY:      '-' (' '|'\n')
        return self.peek(1) in '\0 \t\r\n\x85\u2028\u2029'

    def check_key(self):

        # KEY(flow context):    '?'
        if self.flow_level:
            return True

        # KEY(block context):   '?' (' '|'\n')
        else:
            return self.peek(1) in '\0 \t\r\n\x85\u2028\u2029'

    def check_value(self):

        # VALUE(flow context):  ':'
        if self.flow_level:
            return True

        # VALUE(block context): ':' (' '|'\n')
        else:
            return self.peek(1) in '\0 \t\r\n\x85\u2028\u2029'

    def check_plain(self):

        # A plain scalar may start with any non-space character except:
        #   '-', '?', ':', ',', '[', ']', '{', '}',
        #   '#', '&', '*', '!', '|', '>', '\'', '\"',
        #   '%', '@', '`'.
        #
        # It may also start with
        #   '-', '?', ':'
        # if it is followed by a non-space character.
        #
        # Note that we limit the last rule to the block context (except the
        # '-' character) because we want the flow context to be space
        # independent.
        ch = self.peek()
        return ch not in '\0 \t\r\n\x85\u2028\u2029-?:,[]{}#&*!|>\'\"%@`'  \
                or (self.peek(1) not in '\0 \t\r\n\x85\u2028\u2029'
                        and (ch == '-' or (not self.flow_level and ch in '?:')))

    # Scanners.

    def scan_to_next_token(self):
        # We ignore spaces, line breaks and comments.
        # If we find a line break in the block context, we set the flag
        # `allow_simple_key` on.
        # The byte order mark is stripped if it's the first character in the
        # stream. We do not yet support BOM inside the stream as the
        # specification requires. Any such mark will be considered as a part
        # of the document.
        #
        # TODO: We need to make tab handling rules more sane. A good rule is
        #   Tabs cannot precede tokens
        #   BLOCK-SEQUENCE-START, BLOCK-MAPPING-START, BLOCK-END,
        #   KEY(block), VALUE(block), BLOCK-ENTRY
        # So the checking code is
        #   if <TAB>:
        #       self.allow_simple_keys = False
        # We also need to add the check for `allow_simple_keys == True` to
        # `unwind_indent` before issuing BLOCK-END.
        # Scanners for block, flow, and plain scalars need to be modified.

        if self.index == 0 and self.peek() == '\uFEFF':
            self.forward()
        found = False
        while not found:
            while self.peek() == ' ':
                self.forward()
            if self.peek() == '#':
                while self.peek() not in '\0\r\n\x85\u2028\u2029':
                    self.forward()
            if self.scan_line_break():
                if not self.flow_level:
                    self.allow_simple_key = True
            else:
                found = True

    def scan_directive(self):
        # See the specification for details.
        start_mark = self.get_mark()
        self.forward()
        name = self.scan_directive_name(start_mark)
        value = None
        if name == 'YAML':
            value = self.scan_yaml_directive_value(start_mark)
            end_mark = self.get_mark()
        elif name == 'TAG':
            value = self.scan_tag_directive_value(start_mark)
            end_mark = self.get_mark()
        else:
            end_mark = self.get_mark()
            while self.peek() not in '\0\r\n\x85\u2028\u2029':
                self.forward()
        self.scan_directive_ignored_line(start_mark)
        return DirectiveToken(name, value, start_mark, end_mark)

    def scan_directive_name(self, start_mark):
        # See the specification for details.
        length = 0
        ch = self.peek(length)
        while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'  \
                or ch in '-_':
            length += 1
            ch = self.peek(length)
        if not length:
            raise ScannerError("while scanning a directive", start_mark,
                    "expected alphabetic or numeric character, but found %r"
                    % ch, self.get_mark())
        value = self.prefix(length)
        self.forward(length)
        ch = self.peek()
        if ch not in '\0 \r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a directive", start_mark,
                    "expected alphabetic or numeric character, but found %r"
                    % ch, self.get_mark())
        return value

    def scan_yaml_directive_value(self, start_mark):
        # See the specification for details.
        while self.peek() == ' ':
            self.forward()
        major = self.scan_yaml_directive_number(start_mark)
        if self.peek() != '.':
            raise ScannerError("while scanning a directive", start_mark,
                    "expected a digit or '.', but found %r" % self.peek(),
                    self.get_mark())
        self.forward()
        minor = self.scan_yaml_directive_number(start_mark)
        if self.peek() not in '\0 \r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a directive", start_mark,
                    "expected a digit or ' ', but found %r" % self.peek(),
                    self.get_mark())
        return (major, minor)

    def scan_yaml_directive_number(self, start_mark):
        # See the specification for details.
        ch = self.peek()
        if not ('0' <= ch <= '9'):
            raise ScannerError("while scanning a directive", start_mark,
                    "expected a digit, but found %r" % ch, self.get_mark())
        length = 0
        while '0' <= self.peek(length) <= '9':
            length += 1
        value = int(self.prefix(length))
        self.forward(length)
        return value

    def scan_tag_directive_value(self, start_mark):
        # See the specification for details.
        while self.peek() == ' ':
            self.forward()
        handle = self.scan_tag_directive_handle(start_mark)
        while self.peek() == ' ':
            self.forward()
        prefix = self.scan_tag_directive_prefix(start_mark)
        return (handle, prefix)

    def scan_tag_directive_handle(self, start_mark):
        # See the specification for details.
        value = self.scan_tag_handle('directive', start_mark)
        ch = self.peek()
        if ch != ' ':
            raise ScannerError("while scanning a directive", start_mark,
                    "expected ' ', but found %r" % ch, self.get_mark())
        return value

    def scan_tag_directive_prefix(self, start_mark):
        # See the specification for details.
        value = self.scan_tag_uri('directive', start_mark)
        ch = self.peek()
        if ch not in '\0 \r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a directive", start_mark,
                    "expected ' ', but found %r" % ch, self.get_mark())
        return value

    def scan_directive_ignored_line(self, start_mark):
        # See the specification for details.
        while self.peek() == ' ':
            self.forward()
        if self.peek() == '#':
            while self.peek() not in '\0\r\n\x85\u2028\u2029':
                self.forward()
        ch = self.peek()
        if ch not in '\0\r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a directive", start_mark,
                    "expected a comment or a line break, but found %r"
                        % ch, self.get_mark())
        self.scan_line_break()

    def scan_anchor(self, TokenClass):
        # The specification does not restrict characters for anchors and
        # aliases. This may lead to problems, for instance, the document:
        #   [ *alias, value ]
        # can be interpreted in two ways, as
        #   [ "value" ]
        # and
        #   [ *alias , "value" ]
        # Therefore we restrict aliases to numbers and ASCII letters.
        start_mark = self.get_mark()
        indicator = self.peek()
        if indicator == '*':
            name = 'alias'
        else:
            name = 'anchor'
        self.forward()
        length = 0
        ch = self.peek(length)
        while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'  \
                or ch in '-_':
            length += 1
            ch = self.peek(length)
        if not length:
            raise ScannerError("while scanning an %s" % name, start_mark,
                    "expected alphabetic or numeric character, but found %r"
                    % ch, self.get_mark())
        value = self.prefix(length)
        self.forward(length)
        ch = self.peek()
        if ch not in '\0 \t\r\n\x85\u2028\u2029?:,]}%@`':
            raise ScannerError("while scanning an %s" % name, start_mark,
                    "expected alphabetic or numeric character, but found %r"
                    % ch, self.get_mark())
        end_mark = self.get_mark()
        return TokenClass(value, start_mark, end_mark)

    def scan_tag(self):
        # See the specification for details.
        start_mark = self.get_mark()
        ch = self.peek(1)
        if ch == '<':
            handle = None
            self.forward(2)
            suffix = self.scan_tag_uri('tag', start_mark)
            if self.peek() != '>':
                raise ScannerError("while parsing a tag", start_mark,
                        "expected '>', but found %r" % self.peek(),
                        self.get_mark())
            self.forward()
        elif ch in '\0 \t\r\n\x85\u2028\u2029':
            handle = None
            suffix = '!'
            self.forward()
        else:
            length = 1
            use_handle = False
            while ch not in '\0 \r\n\x85\u2028\u2029':
                if ch == '!':
                    use_handle = True
                    break
                length += 1
                ch = self.peek(length)
            handle = '!'
            if use_handle:
                handle = self.scan_tag_handle('tag', start_mark)
            else:
                handle = '!'
                self.forward()
            suffix = self.scan_tag_uri('tag', start_mark)
        ch = self.peek()
        if ch not in '\0 \r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a tag", start_mark,
                    "expected ' ', but found %r" % ch, self.get_mark())
        value = (handle, suffix)
        end_mark = self.get_mark()
        return TagToken(value, start_mark, end_mark)

    def scan_block_scalar(self, style):
        # See the specification for details.

        if style == '>':
            folded = True
        else:
            folded = False

        chunks = []
        start_mark = self.get_mark()

        # Scan the header.
        self.forward()
        chomping, increment = self.scan_block_scalar_indicators(start_mark)
        self.scan_block_scalar_ignored_line(start_mark)

        # Determine the indentation level and go to the first non-empty line.
        min_indent = self.indent+1
        if min_indent < 1:
            min_indent = 1
        if increment is None:
            breaks, max_indent, end_mark = self.scan_block_scalar_indentation()
            indent = max(min_indent, max_indent)
        else:
            indent = min_indent+increment-1
            breaks, end_mark = self.scan_block_scalar_breaks(indent)
        line_break = ''

        # Scan the inner part of the block scalar.
        while self.column == indent and self.peek() != '\0':
            chunks.extend(breaks)
            leading_non_space = self.peek() not in ' \t'
            length = 0
            while self.peek(length) not in '\0\r\n\x85\u2028\u2029':
                length += 1
            chunks.append(self.prefix(length))
            self.forward(length)
            line_break = self.scan_line_break()
            breaks, end_mark = self.scan_block_scalar_breaks(indent)
            if self.column == indent and self.peek() != '\0':

                # Unfortunately, folding rules are ambiguous.
                #
                # This is the folding according to the specification:
                
                if folded and line_break == '\n'    \
                        and leading_non_space and self.peek() not in ' \t':
                    if not breaks:
                        chunks.append(' ')
                else:
                    chunks.append(line_break)
                
                # This is Clark Evans's interpretation (also in the spec
                # examples):
                #
                #if folded and line_break == '\n':
                #    if not breaks:
                #        if self.peek() not in ' \t':
                #            chunks.append(' ')
                #        else:
                #            chunks.append(line_break)
                #else:
                #    chunks.append(line_break)
            else:
                break

        # Chomp the tail.
        if chomping is not False:
            chunks.append(line_break)
        if chomping is True:
            chunks.extend(breaks)

        # We are done.
        return ScalarToken(''.join(chunks), False, start_mark, end_mark,
                style)

    def scan_block_scalar_indicators(self, start_mark):
        # See the specification for details.
        chomping = None
        increment = None
        ch = self.peek()
        if ch in '+-':
            if ch == '+':
                chomping = True
            else:
                chomping = False
            self.forward()
            ch = self.peek()
            if ch in '0123456789':
                increment = int(ch)
                if increment == 0:
                    raise ScannerError("while scanning a block scalar", start_mark,
                            "expected indentation indicator in the range 1-9, but found 0",
                            self.get_mark())
                self.forward()
        elif ch in '0123456789':
            increment = int(ch)
            if increment == 0:
                raise ScannerError("while scanning a block scalar", start_mark,
                        "expected indentation indicator in the range 1-9, but found 0",
                        self.get_mark())
            self.forward()
            ch = self.peek()
            if ch in '+-':
                if ch == '+':
                    chomping = True
                else:
                    chomping = False
                self.forward()
        ch = self.peek()
        if ch not in '\0 \r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a block scalar", start_mark,
                    "expected chomping or indentation indicators, but found %r"
                    % ch, self.get_mark())
        return chomping, increment

    def scan_block_scalar_ignored_line(self, start_mark):
        # See the specification for details.
        while self.peek() == ' ':
            self.forward()
        if self.peek() == '#':
            while self.peek() not in '\0\r\n\x85\u2028\u2029':
                self.forward()
        ch = self.peek()
        if ch not in '\0\r\n\x85\u2028\u2029':
            raise ScannerError("while scanning a block scalar", start_mark,
                    "expected a comment or a line break, but found %r" % ch,
                    self.get_mark())
        self.scan_line_break()

    def scan_block_scalar_indentation(self):
        # See the specification for details.
        chunks = []
        max_indent = 0
        end_mark = self.get_mark()
        while self.peek() in ' \r\n\x85\u2028\u2029':
            if self.peek() != ' ':
                chunks.append(self.scan_line_break())
                end_mark = self.get_mark()
            else:
                self.forward()
                if self.column > max_indent:
                    max_indent = self.column
        return chunks, max_indent, end_mark

    def scan_block_scalar_breaks(self, indent):
        # See the specification for details.
        chunks = []
        end_mark = self.get_mark()
        while self.column < indent and self.peek() == ' ':
            self.forward()
        while self.peek() in '\r\n\x85\u2028\u2029':
            chunks.append(self.scan_line_break())
            end_mark = self.get_mark()
            while self.column < indent and self.peek() == ' ':
                self.forward()
        return chunks, end_mark

    def scan_flow_scalar(self, style):
        # See the specification for details.
        # Note that we loose indentation rules for quoted scalars. Quoted
        # scalars don't need to adhere indentation because " and ' clearly
        # mark the beginning and the end of them. Therefore we are less
        # restrictive then the specification requires. We only need to check
        # that document separators are not included in scalars.
        if style == '"':
            double = True
        else:
            double = False
        chunks = []
        start_mark = self.get_mark()
        quote = self.peek()
        self.forward()
        chunks.extend(self.scan_flow_scalar_non_spaces(double, start_mark))
        while self.peek() != quote:
            chunks.extend(self.scan_flow_scalar_spaces(double, start_mark))
            chunks.extend(self.scan_flow_scalar_non_spaces(double, start_mark))
        self.forward()
        end_mark = self.get_mark()
        return ScalarToken(''.join(chunks), False, start_mark, end_mark,
                style)

    ESCAPE_REPLACEMENTS = {
        '0':    '\0',
        'a':    '\x07',
        'b':    '\x08',
        't':    '\x09',
        '\t':   '\x09',
        'n':    '\x0A',
        'v':    '\x0B',
        'f':    '\x0C',
        'r':    '\x0D',
        'e':    '\x1B',
        ' ':    '\x20',
        '\"':   '\"',
        '\\':   '\\',
        '/':    '/',
        'N':    '\x85',
        '_':    '\xA0',
        'L':    '\u2028',
        'P':    '\u2029',
    }

    ESCAPE_CODES = {
        'x':    2,
        'u':    4,
        'U':    8,
    }

    def scan_flow_scalar_non_spaces(self, double, start_mark):
        # See the specification for details.
        chunks = []
        while True:
            length = 0
            while self.peek(length) not in '\'\"\\\0 \t\r\n\x85\u2028\u2029':
                length += 1
            if length:
                chunks.append(self.prefix(length))
                self.forward(length)
            ch = self.peek()
            if not double and ch == '\'' and self.peek(1) == '\'':
                chunks.append('\'')
                self.forward(2)
            elif (double and ch == '\'') or (not double and ch in '\"\\'):
                chunks.append(ch)
                self.forward()
            elif double and ch == '\\':
                self.forward()
                ch = self.peek()
                if ch in self.ESCAPE_REPLACEMENTS:
                    chunks.append(self.ESCAPE_REPLACEMENTS[ch])
                    self.forward()
                elif ch in self.ESCAPE_CODES:
                    length = self.ESCAPE_CODES[ch]
                    self.forward()
                    for k in range(length):
                        if self.peek(k) not in '0123456789ABCDEFabcdef':
                            raise ScannerError("while scanning a double-quoted scalar", start_mark,
                                    "expected escape sequence of %d hexadecimal numbers, but found %r" %
                                        (length, self.peek(k)), self.get_mark())
                    code = int(self.prefix(length), 16)
                    chunks.append(chr(code))
                    self.forward(length)
                elif ch in '\r\n\x85\u2028\u2029':
                    self.scan_line_break()
                    chunks.extend(self.scan_flow_scalar_breaks(double, start_mark))
                else:
                    raise ScannerError("while scanning a double-quoted scalar", start_mark,
                            "found unknown escape character %r" % ch, self.get_mark())
            else:
                return chunks

    def scan_flow_scalar_spaces(self, double, start_mark):
        # See the specification for details.
        chunks = []
        length = 0
        while self.peek(length) in ' \t':
            length += 1
        whitespaces = self.prefix(length)
        self.forward(length)
        ch = self.peek()
        if ch == '\0':
            raise ScannerError("while scanning a quoted scalar", start_mark,
                    "found unexpected end of stream", self.get_mark())
        elif ch in '\r\n\x85\u2028\u2029':
            line_break = self.scan_line_break()
            breaks = self.scan_flow_scalar_breaks(double, start_mark)
            if line_break != '\n':
                chunks.append(line_break)
            elif not breaks:
                chunks.append(' ')
            chunks.extend(breaks)
        else:
            chunks.append(whitespaces)
        return chunks

    def scan_flow_scalar_breaks(self, double, start_mark):
        # See the specification for details.
        chunks = []
        while True:
            # Instead of checking indentation, we check for document
            # separators.
            prefix = self.prefix(3)
            if (prefix == '---' or prefix == '...')   \
                    and self.peek(3) in '\0 \t\r\n\x85\u2028\u2029':
                raise ScannerError("while scanning a quoted scalar", start_mark,
                        "found unexpected document separator", self.get_mark())
            while self.peek() in ' \t':
                self.forward()
            if self.peek() in '\r\n\x85\u2028\u2029':
                chunks.append(self.scan_line_break())
            else:
                return chunks

    def scan_plain(self):
        # See the specification for details.
        # We add an additional restriction for the flow context:
        #   plain scalars in the flow context cannot contain ',' or '?'.
        # We also keep track of the `allow_simple_key` flag here.
        # Indentation rules are loosed for the flow context.
        chunks = []
        start_mark = self.get_mark()
        end_mark = start_mark
        indent = self.indent+1
        # We allow zero indentation for scalars, but then we need to check for
        # document separators at the beginning of the line.
        #if indent == 0:
        #    indent = 1
        spaces = []
        while True:
            length = 0
            if self.peek() == '#':
                break
            while True:
                ch = self.peek(length)
                if ch in '\0 \t\r\n\x85\u2028\u2029'    \
                        or (ch == ':' and
                                self.peek(length+1) in '\0 \t\r\n\x85\u2028\u2029'
                                      + (u',[]{}' if self.flow_level else u''))\
                        or (self.flow_level and ch in ',?[]{}'):
                    break
                length += 1
            if length == 0:
                break
            self.allow_simple_key = False
            chunks.extend(spaces)
            chunks.append(self.prefix(length))
            self.forward(length)
            end_mark = self.get_mark()
            spaces = self.scan_plain_spaces(indent, start_mark)
            if not spaces or self.peek() == '#' \
                    or (not self.flow_level and self.column < indent):
                break
        return ScalarToken(''.join(chunks), True, start_mark, end_mark)

    def scan_plain_spaces(self, indent, start_mark):
        # See the specification for details.
        # The specification is really confusing about tabs in plain scalars.
        # We just forbid them completely. Do not use tabs in YAML!
        chunks = []
        length = 0
        while self.peek(length) in ' ':
            length += 1
        whitespaces = self.prefix(length)
        self.forward(length)
        ch = self.peek()
        if ch in '\r\n\x85\u2028\u2029':
            line_break = self.scan_line_break()
            self.allow_simple_key = True
            prefix = self.prefix(3)
            if (prefix == '---' or prefix == '...')   \
                    and self.peek(3) in '\0 \t\r\n\x85\u2028\u2029':
                return
            breaks = []
            while self.peek() in ' \r\n\x85\u2028\u2029':
                if self.peek() == ' ':
                    self.forward()
                else:
                    breaks.append(self.scan_line_break())
                    prefix = self.prefix(3)
                    if (prefix == '---' or prefix == '...')   \
                            and self.peek(3) in '\0 \t\r\n\x85\u2028\u2029':
                        return
            if line_break != '\n':
                chunks.append(line_break)
            elif not breaks:
                chunks.append(' ')
            chunks.extend(breaks)
        elif whitespaces:
            chunks.append(whitespaces)
        return chunks

    def scan_tag_handle(self, name, start_mark):
        # See the specification for details.
        # For some strange reasons, the specification does not allow '_' in
        # tag handles. I have allowed it anyway.
        ch = self.peek()
        if ch != '!':
            raise ScannerError("while scanning a %s" % name, start_mark,
                    "expected '!', but found %r" % ch, self.get_mark())
        length = 1
        ch = self.peek(length)
        if ch != ' ':
            while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'  \
                    or ch in '-_':
                length += 1
                ch = self.peek(length)
            if ch != '!':
                self.forward(length)
                raise ScannerError("while scanning a %s" % name, start_mark,
                        "expected '!', but found %r" % ch, self.get_mark())
            length += 1
        value = self.prefix(length)
        self.forward(length)
        return value

    def scan_tag_uri(self, name, start_mark):
        # See the specification for details.
        # Note: we do not check if URI is well-formed.
        chunks = []
        length = 0
        ch = self.peek(length)
        while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'  \
                or ch in '-;/?:@&=+$,_.!~*\'()[]%':
            if ch == '%':
                chunks.append(self.prefix(length))
                self.forward(length)
                length = 0
                chunks.append(self.scan_uri_escapes(name, start_mark))
            else:
                length += 1
            ch = self.peek(length)
        if length:
            chunks.append(self.prefix(length))
            self.forward(length)
            length = 0
        if not chunks:
            raise ScannerError("while parsing a %s" % name, start_mark,
                    "expected URI, but found %r" % ch, self.get_mark())
        return ''.join(chunks)

    def scan_uri_escapes(self, name, start_mark):
        # See the specification for details.
        codes = []
        mark = self.get_mark()
        while self.peek() == '%':
            self.forward()
            for k in range(2):
                if self.peek(k) not in '0123456789ABCDEFabcdef':
                    raise ScannerError("while scanning a %s" % name, start_mark,
                            "expected URI escape sequence of 2 hexadecimal numbers, but found %r"
                            % self.peek(k), self.get_mark())
            codes.append(int(self.prefix(2), 16))
            self.forward(2)
        try:
            value = bytes(codes).decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ScannerError("while scanning a %s" % name, start_mark, str(exc), mark)
        return value

    def scan_line_break(self):
        # Transforms:
        #   '\r\n'      :   '\n'
        #   '\r'        :   '\n'
        #   '\n'        :   '\n'
        #   '\x85'      :   '\n'
        #   '\u2028'    :   '\u2028'
        #   '\u2029     :   '\u2029'
        #   default     :   ''
        ch = self.peek()
        if ch in '\r\n\x85':
            if self.prefix(2) == '\r\n':
                self.forward(2)
            else:
                self.forward()
            return '\n'
        elif ch in '\u2028\u2029':
            self.forward()
            return ch
        return ''
<|endoftext|>
from .error import *

from .tokens import *
from .events import *
from .nodes import *

from .loader import *
from .dumper import *

__version__ = '6.0.3'
try:
    from .cyaml import *
    __with_libyaml__ = True
except ImportError:
    __with_libyaml__ = False

import io

#------------------------------------------------------------------------------
# XXX "Warnings control" is now deprecated. Leaving in the API function to not
# break code that uses it.
#------------------------------------------------------------------------------
def warnings(settings=None):
    if settings is None:
        return {}

#------------------------------------------------------------------------------
def scan(stream, Loader=Loader):
    """
    Scan a YAML stream and produce scanning tokens.
    """
    loader = Loader(stream)
    try:
        while loader.check_token():
            yield loader.get_token()
    finally:
        loader.dispose()

def parse(stream, Loader=Loader):
    """
    Parse a YAML stream and produce parsing events.
    """
    loader = Loader(stream)
    try:
        while loader.check_event():
            yield loader.get_event()
    finally:
        loader.dispose()

def compose(stream, Loader=Loader):
    """
    Parse the first YAML document in a stream
    and produce the corresponding representation tree.
    """
    loader = Loader(stream)
    try:
        return loader.get_single_node()
    finally:
        loader.dispose()

def compose_all(stream, Loader=Loader):
    """
    Parse all YAML documents in a stream
    and produce corresponding representation trees.
    """
    loader = Loader(stream)
    try:
        while loader.check_node():
            yield loader.get_node()
    finally:
        loader.dispose()

def load(stream, Loader):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    """
    loader = Loader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()

def load_all(stream, Loader):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.
    """
    loader = Loader(stream)
    try:
        while loader.check_data():
            yield loader.get_data()
    finally:
        loader.dispose()

def full_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve all tags except those known to be
    unsafe on untrusted input.
    """
    return load(stream, FullLoader)

def full_load_all(stream):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.

    Resolve all tags except those known to be
    unsafe on untrusted input.
    """
    return load_all(stream, FullLoader)

def safe_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve only basic YAML tags. This is known
    to be safe for untrusted input.
    """
    return load(stream, SafeLoader)

def safe_load_all(stream):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.

    Resolve only basic YAML tags. This is known
    to be safe for untrusted input.
    """
    return load_all(stream, SafeLoader)

def unsafe_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.

    Resolve all tags, even those known to be
    unsafe on untrusted input.
    """
    return load(stream, UnsafeLoader)

def unsafe_load_all(stream):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.

    Resolve all tags, even those known to be
    unsafe on untrusted input.
    """
    return load_all(stream, UnsafeLoader)

def emit(events, stream=None, Dumper=Dumper,
        canonical=None, indent=None, width=None,
        allow_unicode=None, line_break=None):
    """
    Emit YAML parsing events into a stream.
    If stream is None, return the produced string instead.
    """
    getvalue = None
    if stream is None:
        stream = io.StringIO()
        getvalue = stream.getvalue
    dumper = Dumper(stream, canonical=canonical, indent=indent, width=width,
            allow_unicode=allow_unicode, line_break=line_break)
    try:
        for event in events:
            dumper.emit(event)
    finally:
        dumper.dispose()
    if getvalue:
        return getvalue()

def serialize_all(nodes, stream=None, Dumper=Dumper,
        canonical=None, indent=None, width=None,
        allow_unicode=None, line_break=None,
        encoding=None, explicit_start=None, explicit_end=None,
        version=None, tags=None):
    """
    Serialize a sequence of representation trees into a YAML stream.
    If stream is None, return the produced string instead.
    """
    getvalue = None
    if stream is None:
        if encoding is None:
            stream = io.StringIO()
        else:
            stream = io.BytesIO()
        getvalue = stream.getvalue
    dumper = Dumper(stream, canonical=canonical, indent=indent, width=width,
            allow_unicode=allow_unicode, line_break=line_break,
            encoding=encoding, version=version, tags=tags,
            explicit_start=explicit_start, explicit_end=explicit_end)
    try:
        dumper.open()
        for node in nodes:
            dumper.serialize(node)
        dumper.close()
    finally:
        dumper.dispose()
    if getvalue:
        return getvalue()

def serialize(node, stream=None, Dumper=Dumper, **kwds):
    """
    Serialize a representation tree into a YAML stream.
    If stream is None, return the produced string instead.
    """
    return serialize_all([node], stream, Dumper=Dumper, **kwds)

def dump_all(documents, stream=None, Dumper=Dumper,
        default_style=None, default_flow_style=False,
        canonical=None, indent=None, width=None,
        allow_unicode=None, line_break=None,
        encoding=None, explicit_start=None, explicit_end=None,
        version=None, tags=None, sort_keys=True):
    """
    Serialize a sequence of Python objects into a YAML stream.
    If stream is None, return the produced string instead.
    """
    getvalue = None
    if stream is None:
        if encoding is None:
            stream = io.StringIO()
        else:
            stream = io.BytesIO()
        getvalue = stream.getvalue
    dumper = Dumper(stream, default_style=default_style,
            default_flow_style=default_flow_style,
            canonical=canonical, indent=indent, width=width,
            allow_unicode=allow_unicode, line_break=line_break,
            encoding=encoding, version=version, tags=tags,
            explicit_start=explicit_start, explicit_end=explicit_end, sort_keys=sort_keys)
    try:
        dumper.open()
        for data in documents:
            dumper.represent(data)
        dumper.close()
    finally:
        dumper.dispose()
    if getvalue:
        return getvalue()

def dump(data, stream=None, Dumper=Dumper, **kwds):
    """
    Serialize a Python object into a YAML stream.
    If stream is None, return the produced string instead.
    """
    return dump_all([data], stream, Dumper=Dumper, **kwds)

def safe_dump_all(documents, stream=None, **kwds):
    """
    Serialize a sequence of Python objects into a YAML stream.
    Produce only basic YAML tags.
    If stream is None, return the produced string instead.
    """
    return dump_all(documents, stream, Dumper=SafeDumper, **kwds)

def safe_dump(data, stream=None, **kwds):
    """
    Serialize a Python object into a YAML stream.
    Produce only basic YAML tags.
    If stream is None, return the produced string instead.
    """
    return dump_all([data], stream, Dumper=SafeDumper, **kwds)

def add_implicit_resolver(tag, regexp, first=None,
        Loader=None, Dumper=Dumper):
    """
    Add an implicit scalar detector.
    If an implicit scalar value matches the given regexp,
    the corresponding tag is assigned to the scalar.
    first is a sequence of possible initial characters or None.
    """
    if Loader is None:
        loader.Loader.add_implicit_resolver(tag, regexp, first)
        loader.FullLoader.add_implicit_resolver(tag, regexp, first)
        loader.UnsafeLoader.add_implicit_resolver(tag, regexp, first)
    else:
        Loader.add_implicit_resolver(tag, regexp, first)
    Dumper.add_implicit_resolver(tag, regexp, first)

def add_path_resolver(tag, path, kind=None, Loader=None, Dumper=Dumper):
    """
    Add a path based resolver for the given tag.
    A path is a list of keys that forms a path
    to a node in the representation tree.
    Keys can be string values, integers, or None.
    """
    if Loader is None:
        loader.Loader.add_path_resolver(tag, path, kind)
        loader.FullLoader.add_path_resolver(tag, path, kind)
        loader.UnsafeLoader.add_path_resolver(tag, path, kind)
    else:
        Loader.add_path_resolver(tag, path, kind)
    Dumper.add_path_resolver(tag, path, kind)

def add_constructor(tag, constructor, Loader=None):
    """
    Add a constructor for the given tag.
    Constructor is a function that accepts a Loader instance
    and a node object and produces the corresponding Python object.
    """
    if Loader is None:
        loader.Loader.add_constructor(tag, constructor)
        loader.FullLoader.add_constructor(tag, constructor)
        loader.UnsafeLoader.add_constructor(tag, constructor)
    else:
        Loader.add_constructor(tag, constructor)

def add_multi_constructor(tag_prefix, multi_constructor, Loader=None):
    """
    Add a multi-constructor for the given tag prefix.
    Multi-constructor is called for a node if its tag starts with tag_prefix.
    Multi-constructor accepts a Loader instance, a tag suffix,
    and a node object and produces the corresponding Python object.
    """
    if Loader is None:
        loader.Loader.add_multi_constructor(tag_prefix, multi_constructor)
        loader.FullLoader.add_multi_constructor(tag_prefix, multi_constructor)
        loader.UnsafeLoader.add_multi_constructor(tag_prefix, multi_constructor)
    else:
        Loader.add_multi_constructor(tag_prefix, multi_constructor)

def add_representer(data_type, representer, Dumper=Dumper):
    """
    Add a representer for the given type.
    Representer is a function accepting a Dumper instance
    and an instance of the given data type
    and producing the corresponding representation node.
    """
    Dumper.add_representer(data_type, representer)

def add_multi_representer(data_type, multi_representer, Dumper=Dumper):
    """
    Add a representer for the given type.
    Multi-representer is a function accepting a Dumper instance
    and an instance of the given data type or subtype
    and producing the corresponding representation node.
    """
    Dumper.add_multi_representer(data_type, multi_representer)

class YAMLObjectMetaclass(type):
    """
    The metaclass for YAMLObject.
    """
    def __init__(cls, name, bases, kwds):
        super(YAMLObjectMetaclass, cls).__init__(name, bases, kwds)
        if 'yaml_tag' in kwds and kwds['yaml_tag'] is not None:
            if isinstance(cls.yaml_loader, list):
                for loader in cls.yaml_loader:
                    loader.add_constructor(cls.yaml_tag, cls.from_yaml)
            else:
                cls.yaml_loader.add_constructor(cls.yaml_tag, cls.from_yaml)

            cls.yaml_dumper.add_representer(cls, cls.to_yaml)

class YAMLObject(metaclass=YAMLObjectMetaclass):
    """
    An object that can dump itself to a YAML stream
    and load itself from a YAML stream.
    """

    __slots__ = ()  # no direct instantiation, so allow immutable subclasses

    yaml_loader = [Loader, FullLoader, UnsafeLoader]
    yaml_dumper = Dumper

    yaml_tag = None
    yaml_flow_style = None

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Convert a representation node to a Python object.
        """
        return loader.construct_yaml_object(node, cls)

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Convert a Python object to a representation node.
        """
        return dumper.represent_yaml_object(cls.yaml_tag, data, cls,
                flow_style=cls.yaml_flow_style)

<|endoftext|>
class Node(object):
    def __init__(self, tag, value, start_mark, end_mark):
        self.tag = tag
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark
    def __repr__(self):
        value = self.value
        #if isinstance(value, list):
        #    if len(value) == 0:
        #        value = '<empty>'
        #    elif len(value) == 1:
        #        value = '<1 item>'
        #    else:
        #        value = '<%d items>' % len(value)
        #else:
        #    if len(value) > 75:
        #        value = repr(value[:70]+u' ... ')
        #    else:
        #        value = repr(value)
        value = repr(value)
        return '%s(tag=%r, value=%s)' % (self.__class__.__name__, self.tag, value)

class ScalarNode(Node):
    id = 'scalar'
    def __init__(self, tag, value,
            start_mark=None, end_mark=None, style=None):
        self.tag = tag
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.style = style

class CollectionNode(Node):
    def __init__(self, tag, value,
            start_mark=None, end_mark=None, flow_style=None):
        self.tag = tag
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.flow_style = flow_style

class SequenceNode(CollectionNode):
    id = 'sequence'

class MappingNode(CollectionNode):
    id = 'mapping'

<|endoftext|>
__all__ = ['Serializer', 'SerializerError']

from .error import YAMLError
from .events import *
from .nodes import *

class SerializerError(YAMLError):
    pass

class Serializer:

    ANCHOR_TEMPLATE = 'id%03d'

    def __init__(self, encoding=None,
            explicit_start=None, explicit_end=None, version=None, tags=None):
        self.use_encoding = encoding
        self.use_explicit_start = explicit_start
        self.use_explicit_end = explicit_end
        self.use_version = version
        self.use_tags = tags
        self.serialized_nodes = {}
        self.anchors = {}
        self.last_anchor_id = 0
        self.closed = None

    def open(self):
        if self.closed is None:
            self.emit(StreamStartEvent(encoding=self.use_encoding))
            self.closed = False
        elif self.closed:
            raise SerializerError("serializer is closed")
        else:
            raise SerializerError("serializer is already opened")

    def close(self):
        if self.closed is None:
            raise SerializerError("serializer is not opened")
        elif not self.closed:
            self.emit(StreamEndEvent())
            self.closed = True

    #def __del__(self):
    #    self.close()

    def serialize(self, node):
        if self.closed is None:
            raise SerializerError("serializer is not opened")
        elif self.closed:
            raise SerializerError("serializer is closed")
        self.emit(DocumentStartEvent(explicit=self.use_explicit_start,
            version=self.use_version, tags=self.use_tags))
        self.anchor_node(node)
        self.serialize_node(node, None, None)
        self.emit(DocumentEndEvent(explicit=self.use_explicit_end))
        self.serialized_nodes = {}
        self.anchors = {}
        self.last_anchor_id = 0

    def anchor_node(self, node):
        if node in self.anchors:
            if self.anchors[node] is None:
                self.anchors[node] = self.generate_anchor(node)
        else:
            self.anchors[node] = None
            if isinstance(node, SequenceNode):
                for item in node.value:
                    self.anchor_node(item)
            elif isinstance(node, MappingNode):
                for key, value in node.value:
                    self.anchor_node(key)
                    self.anchor_node(value)

    def generate_anchor(self, node):
        self.last_anchor_id += 1
        return self.ANCHOR_TEMPLATE % self.last_anchor_id

    def serialize_node(self, node, parent, index):
        alias = self.anchors[node]
        if node in self.serialized_nodes:
            self.emit(AliasEvent(alias))
        else:
            self.serialized_nodes[node] = True
            self.descend_resolver(parent, index)
            if isinstance(node, ScalarNode):
                detected_tag = self.resolve(ScalarNode, node.value, (True, False))
                default_tag = self.resolve(ScalarNode, node.value, (False, True))
                implicit = (node.tag == detected_tag), (node.tag == default_tag)
                self.emit(ScalarEvent(alias, node.tag, implicit, node.value,
                    style=node.style))
            elif isinstance(node, SequenceNode):
                implicit = (node.tag
                            == self.resolve(SequenceNode, node.value, True))
                self.emit(SequenceStartEvent(alias, node.tag, implicit,
                    flow_style=node.flow_style))
                index = 0
                for item in node.value:
                    self.serialize_node(item, node, index)
                    index += 1
                self.emit(SequenceEndEvent())
            elif isinstance(node, MappingNode):
                implicit = (node.tag
                            == self.resolve(MappingNode, node.value, True))
                self.emit(MappingStartEvent(alias, node.tag, implicit,
                    flow_style=node.flow_style))
                for key, value in node.value:
                    self.serialize_node(key, node, None)
                    self.serialize_node(value, node, key)
                self.emit(MappingEndEvent())
            self.ascend_resolver()

<|endoftext|>
__all__ = ['BaseRepresenter', 'SafeRepresenter', 'Representer',
    'RepresenterError']

from .error import *
from .nodes import *

import datetime, copyreg, types, base64, collections

class RepresenterError(YAMLError):
    pass

class BaseRepresenter:

    yaml_representers = {}
    yaml_multi_representers = {}

    def __init__(self, default_style=None, default_flow_style=False, sort_keys=True):
        self.default_style = default_style
        self.sort_keys = sort_keys
        self.default_flow_style = default_flow_style
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None

    def represent(self, data):
        node = self.represent_data(data)
        self.serialize(node)
        self.represented_objects = {}
        self.object_keeper = []
        self.alias_key = None

    def represent_data(self, data):
        if self.ignore_aliases(data):
            self.alias_key = None
        else:
            self.alias_key = id(data)
        if self.alias_key is not None:
            if self.alias_key in self.represented_objects:
                node = self.represented_objects[self.alias_key]
                #if node is None:
                #    raise RepresenterError("recursive objects are not allowed: %r" % data)
                return node
            #self.represented_objects[alias_key] = None
            self.object_keeper.append(data)
        data_types = type(data).__mro__
        if data_types[0] in self.yaml_representers:
            node = self.yaml_representers[data_types[0]](self, data)
        else:
            for data_type in data_types:
                if data_type in self.yaml_multi_representers:
                    node = self.yaml_multi_representers[data_type](self, data)
                    break
            else:
                if None in self.yaml_multi_representers:
                    node = self.yaml_multi_representers[None](self, data)
                elif None in self.yaml_representers:
                    node = self.yaml_representers[None](self, data)
                else:
                    node = ScalarNode(None, str(data))
        #if alias_key is not None:
        #    self.represented_objects[alias_key] = node
        return node

    @classmethod
    def add_representer(cls, data_type, representer):
        if not 'yaml_representers' in cls.__dict__:
            cls.yaml_representers = cls.yaml_representers.copy()
        cls.yaml_representers[data_type] = representer

    @classmethod
    def add_multi_representer(cls, data_type, representer):
        if not 'yaml_multi_representers' in cls.__dict__:
            cls.yaml_multi_representers = cls.yaml_multi_representers.copy()
        cls.yaml_multi_representers[data_type] = representer

    def represent_scalar(self, tag, value, style=None):
        if style is None:
            style = self.default_style
        node = ScalarNode(tag, value, style=style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        return node

    def represent_sequence(self, tag, sequence, flow_style=None):
        value = []
        node = SequenceNode(tag, value, flow_style=flow_style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        for item in sequence:
            node_item = self.represent_data(item)
            if not (isinstance(node_item, ScalarNode) and not node_item.style):
                best_style = False
            value.append(node_item)
        if flow_style is None:
            if self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        return node

    def represent_mapping(self, tag, mapping, flow_style=None):
        value = []
        node = MappingNode(tag, value, flow_style=flow_style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node
        best_style = True
        if hasattr(mapping, 'items'):
            mapping = list(mapping.items())
            if self.sort_keys:
                try:
                    mapping = sorted(mapping)
                except TypeError:
                    pass
        for item_key, item_value in mapping:
            node_key = self.represent_data(item_key)
            node_value = self.represent_data(item_value)
            if not (isinstance(node_key, ScalarNode) and not node_key.style):
                best_style = False
            if not (isinstance(node_value, ScalarNode) and not node_value.style):
                best_style = False
            value.append((node_key, node_value))
        if flow_style is None:
            if self.default_flow_style is not None:
                node.flow_style = self.default_flow_style
            else:
                node.flow_style = best_style
        return node

    def ignore_aliases(self, data):
        return False

class SafeRepresenter(BaseRepresenter):

    def ignore_aliases(self, data):
        if data is None:
            return True
        if isinstance(data, tuple) and data == ():
            return True
        if isinstance(data, (str, bytes, bool, int, float)):
            return True

    def represent_none(self, data):
        return self.represent_scalar('tag:yaml.org,2002:null', 'null')

    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data)

    def represent_binary(self, data):
        if hasattr(base64, 'encodebytes'):
            data = base64.encodebytes(data).decode('ascii')
        else:
            data = base64.encodestring(data).decode('ascii')
        return self.represent_scalar('tag:yaml.org,2002:binary', data, style='|')

    def represent_bool(self, data):
        if data:
            value = 'true'
        else:
            value = 'false'
        return self.represent_scalar('tag:yaml.org,2002:bool', value)

    def represent_int(self, data):
        return self.represent_scalar('tag:yaml.org,2002:int', str(data))

    inf_value = 1e300
    while repr(inf_value) != repr(inf_value*inf_value):
        inf_value *= inf_value

    def represent_float(self, data):
        if data != data or (data == 0.0 and data == 1.0):
            value = '.nan'
        elif data == self.inf_value:
            value = '.inf'
        elif data == -self.inf_value:
            value = '-.inf'
        else:
            value = repr(data).lower()
            # Note that in some cases `repr(data)` represents a float number
            # without the decimal parts.  For instance:
            #   >>> repr(1e17)
            #   '1e17'
            # Unfortunately, this is not a valid float representation according
            # to the definition of the `!!float` tag.  We fix this by adding
            # '.0' before the 'e' symbol.
            if '.' not in value and 'e' in value:
                value = value.replace('e', '.0e', 1)
        return self.represent_scalar('tag:yaml.org,2002:float', value)

    def represent_list(self, data):
        #pairs = (len(data) > 0 and isinstance(data, list))
        #if pairs:
        #    for item in data:
        #        if not isinstance(item, tuple) or len(item) != 2:
        #            pairs = False
        #            break
        #if not pairs:
            return self.represent_sequence('tag:yaml.org,2002:seq', data)
        #value = []
        #for item_key, item_value in data:
        #    value.append(self.represent_mapping(u'tag:yaml.org,2002:map',
        #        [(item_key, item_value)]))
        #return SequenceNode(u'tag:yaml.org,2002:pairs', value)

    def represent_dict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data)

    def represent_set(self, data):
        value = {}
        for key in data:
            value[key] = None
        return self.represent_mapping('tag:yaml.org,2002:set', value)

    def represent_date(self, data):
        value = data.isoformat()
        return self.represent_scalar('tag:yaml.org,2002:timestamp', value)

    def represent_datetime(self, data):
        value = data.isoformat(' ')
        return self.represent_scalar('tag:yaml.org,2002:timestamp', value)

    def represent_yaml_object(self, tag, data, cls, flow_style=None):
        if hasattr(data, '__getstate__'):
            state = data.__getstate__()
        else:
            state = data.__dict__.copy()
        return self.represent_mapping(tag, state, flow_style=flow_style)

    def represent_undefined(self, data):
        raise RepresenterError("cannot represent an object", data)

SafeRepresenter.add_representer(type(None),
        SafeRepresenter.represent_none)

SafeRepresenter.add_representer(str,
        SafeRepresenter.represent_str)

SafeRepresenter.add_representer(bytes,
        SafeRepresenter.represent_binary)

SafeRepresenter.add_representer(bool,
        SafeRepresenter.represent_bool)

SafeRepresenter.add_representer(int,
        SafeRepresenter.represent_int)

SafeRepresenter.add_representer(float,
        SafeRepresenter.represent_float)

SafeRepresenter.add_representer(list,
        SafeRepresenter.represent_list)

SafeRepresenter.add_representer(tuple,
        SafeRepresenter.represent_list)

SafeRepresenter.add_representer(dict,
        SafeRepresenter.represent_dict)

SafeRepresenter.add_representer(set,
        SafeRepresenter.represent_set)

SafeRepresenter.add_representer(datetime.date,
        SafeRepresenter.represent_date)

SafeRepresenter.add_representer(datetime.datetime,
        SafeRepresenter.represent_datetime)

SafeRepresenter.add_representer(None,
        SafeRepresenter.represent_undefined)

class Representer(SafeRepresenter):

    def represent_complex(self, data):
        if data.imag == 0.0:
            data = '%r' % data.real
        elif data.real == 0.0:
            data = '%rj' % data.imag
        elif data.imag > 0:
            data = '%r+%rj' % (data.real, data.imag)
        else:
            data = '%r%rj' % (data.real, data.imag)
        return self.represent_scalar('tag:yaml.org,2002:python/complex', data)

    def represent_tuple(self, data):
        return self.represent_sequence('tag:yaml.org,2002:python/tuple', data)

    def represent_name(self, data):
        name = '%s.%s' % (data.__module__, data.__name__)
        return self.represent_scalar('tag:yaml.org,2002:python/name:'+name, '')

    def represent_module(self, data):
        return self.represent_scalar(
                'tag:yaml.org,2002:python/module:'+data.__name__, '')

    def represent_object(self, data):
        # We use __reduce__ API to save the data. data.__reduce__ returns
        # a tuple of length 2-5:
        #   (function, args, state, listitems, dictitems)

        # For reconstructing, we calls function(*args), then set its state,
        # listitems, and dictitems if they are not None.

        # A special case is when function.__name__ == '__newobj__'. In this
        # case we create the object with args[0].__new__(*args).

        # Another special case is when __reduce__ returns a string - we don't
        # support it.

        # We produce a !!python/object, !!python/object/new or
        # !!python/object/apply node.

        cls = type(data)
        if cls in copyreg.dispatch_table:
            reduce = copyreg.dispatch_table[cls](data)
        elif hasattr(data, '__reduce_ex__'):
            reduce = data.__reduce_ex__(2)
        elif hasattr(data, '__reduce__'):
            reduce = data.__reduce__()
        else:
            raise RepresenterError("cannot represent an object", data)
        reduce = (list(reduce)+[None]*5)[:5]
        function, args, state, listitems, dictitems = reduce
        args = list(args)
        if state is None:
            state = {}
        if listitems is not None:
            listitems = list(listitems)
        if dictitems is not None:
            dictitems = dict(dictitems)
        if function.__name__ == '__newobj__':
            function = args[0]
            args = args[1:]
            tag = 'tag:yaml.org,2002:python/object/new:'
            newobj = True
        else:
            tag = 'tag:yaml.org,2002:python/object/apply:'
            newobj = False
        function_name = '%s.%s' % (function.__module__, function.__name__)
        if not args and not listitems and not dictitems \
                and isinstance(state, dict) and newobj:
            return self.represent_mapping(
                    'tag:yaml.org,2002:python/object:'+function_name, state)
        if not listitems and not dictitems  \
                and isinstance(state, dict) and not state:
            return self.represent_sequence(tag+function_name, args)
        value = {}
        if args:
            value['args'] = args
        if state or not isinstance(state, dict):
            value['state'] = state
        if listitems:
            value['listitems'] = listitems
        if dictitems:
            value['dictitems'] = dictitems
        return self.represent_mapping(tag+function_name, value)

    def represent_ordered_dict(self, data):
        # Provide uniform representation across different Python versions.
        data_type = type(data)
        tag = 'tag:yaml.org,2002:python/object/apply:%s.%s' \
                % (data_type.__module__, data_type.__name__)
        items = [[key, value] for key, value in data.items()]
        return self.represent_sequence(tag, [items])

Representer.add_representer(complex,
        Representer.represent_complex)

Representer.add_representer(tuple,
        Representer.represent_tuple)

Representer.add_multi_representer(type,
        Representer.represent_name)

Representer.add_representer(collections.OrderedDict,
        Representer.represent_ordered_dict)

Representer.add_representer(types.FunctionType,
        Representer.represent_name)

Representer.add_representer(types.BuiltinFunctionType,
        Representer.represent_name)

Representer.add_representer(types.ModuleType,
        Representer.represent_module)

Representer.add_multi_representer(object,
        Representer.represent_object)

<|endoftext|>
__all__ = ['BaseDumper', 'SafeDumper', 'Dumper']

from .emitter import *
from .serializer import *
from .representer import *
from .resolver import *

class BaseDumper(Emitter, Serializer, BaseRepresenter, BaseResolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        Emitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width,
                allow_unicode=allow_unicode, line_break=line_break)
        Serializer.__init__(self, encoding=encoding,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        Representer.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

class SafeDumper(Emitter, Serializer, SafeRepresenter, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        Emitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width,
                allow_unicode=allow_unicode, line_break=line_break)
        Serializer.__init__(self, encoding=encoding,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        SafeRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

class Dumper(Emitter, Serializer, Representer, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        Emitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width,
                allow_unicode=allow_unicode, line_break=line_break)
        Serializer.__init__(self, encoding=encoding,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        Representer.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

<|endoftext|>
__all__ = ['Mark', 'YAMLError', 'MarkedYAMLError']

class Mark:

    def __init__(self, name, index, line, column, buffer, pointer):
        self.name = name
        self.index = index
        self.line = line
        self.column = column
        self.buffer = buffer
        self.pointer = pointer

    def get_snippet(self, indent=4, max_length=75):
        if self.buffer is None:
            return None
        head = ''
        start = self.pointer
        while start > 0 and self.buffer[start-1] not in '\0\r\n\x85\u2028\u2029':
            start -= 1
            if self.pointer-start > max_length/2-1:
                head = ' ... '
                start += 5
                break
        tail = ''
        end = self.pointer
        while end < len(self.buffer) and self.buffer[end] not in '\0\r\n\x85\u2028\u2029':
            end += 1
            if end-self.pointer > max_length/2-1:
                tail = ' ... '
                end -= 5
                break
        snippet = self.buffer[start:end]
        return ' '*indent + head + snippet + tail + '\n'  \
                + ' '*(indent+self.pointer-start+len(head)) + '^'

    def __str__(self):
        snippet = self.get_snippet()
        where = "  in \"%s\", line %d, column %d"   \
                % (self.name, self.line+1, self.column+1)
        if snippet is not None:
            where += ":\n"+snippet
        return where

class YAMLError(Exception):
    pass

class MarkedYAMLError(YAMLError):

    def __init__(self, context=None, context_mark=None,
            problem=None, problem_mark=None, note=None):
        self.context = context
        self.context_mark = context_mark
        self.problem = problem
        self.problem_mark = problem_mark
        self.note = note

    def __str__(self):
        lines = []
        if self.context is not None:
            lines.append(self.context)
        if self.context_mark is not None  \
            and (self.problem is None or self.problem_mark is None
                    or self.context_mark.name != self.problem_mark.name
                    or self.context_mark.line != self.problem_mark.line
                    or self.context_mark.column != self.problem_mark.column):
            lines.append(str(self.context_mark))
        if self.problem is not None:
            lines.append(self.problem)
        if self.problem_mark is not None:
            lines.append(str(self.problem_mark))
        if self.note is not None:
            lines.append(self.note)
        return '\n'.join(lines)

<|endoftext|>
# Emitter expects events obeying the following grammar:
# stream ::= STREAM-START document* STREAM-END
# document ::= DOCUMENT-START node DOCUMENT-END
# node ::= SCALAR | sequence | mapping
# sequence ::= SEQUENCE-START node* SEQUENCE-END
# mapping ::= MAPPING-START (node node)* MAPPING-END

__all__ = ['Emitter', 'EmitterError']

from .error import YAMLError
from .events import *

class EmitterError(YAMLError):
    pass

class ScalarAnalysis:
    def __init__(self, scalar, empty, multiline,
            allow_flow_plain, allow_block_plain,
            allow_single_quoted, allow_double_quoted,
            allow_block):
        self.scalar = scalar
        self.empty = empty
        self.multiline = multiline
        self.allow_flow_plain = allow_flow_plain
        self.allow_block_plain = allow_block_plain
        self.allow_single_quoted = allow_single_quoted
        self.allow_double_quoted = allow_double_quoted
        self.allow_block = allow_block

class Emitter:

    DEFAULT_TAG_PREFIXES = {
        '!' : '!',
        'tag:yaml.org,2002:' : '!!',
    }

    def __init__(self, stream, canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None):

        # The stream should have the methods `write` and possibly `flush`.
        self.stream = stream

        # Encoding can be overridden by STREAM-START.
        self.encoding = None

        # Emitter is a state machine with a stack of states to handle nested
        # structures.
        self.states = []
        self.state = self.expect_stream_start

        # Current event and the event queue.
        self.events = []
        self.event = None

        # The current indentation level and the stack of previous indents.
        self.indents = []
        self.indent = None

        # Flow level.
        self.flow_level = 0

        # Contexts.
        self.root_context = False
        self.sequence_context = False
        self.mapping_context = False
        self.simple_key_context = False

        # Characteristics of the last emitted character:
        #  - current position.
        #  - is it a whitespace?
        #  - is it an indention character
        #    (indentation space, '-', '?', or ':')?
        self.line = 0
        self.column = 0
        self.whitespace = True
        self.indention = True

        # Whether the document requires an explicit document indicator
        self.open_ended = False

        # Formatting details.
        self.canonical = canonical
        self.allow_unicode = allow_unicode
        self.best_indent = 2
        if indent and 1 < indent < 10:
            self.best_indent = indent
        self.best_width = 80
        if width and width > self.best_indent*2:
            self.best_width = width
        self.best_line_break = '\n'
        if line_break in ['\r', '\n', '\r\n']:
            self.best_line_break = line_break

        # Tag prefixes.
        self.tag_prefixes = None

        # Prepared anchor and tag.
        self.prepared_anchor = None
        self.prepared_tag = None

        # Scalar analysis and style.
        self.analysis = None
        self.style = None

    def dispose(self):
        # Reset the state attributes (to clear self-references)
        self.states = []
        self.state = None

    def emit(self, event):
        self.events.append(event)
        while not self.need_more_events():
            self.event = self.events.pop(0)
            self.state()
            self.event = None

    # In some cases, we wait for a few next events before emitting.

    def need_more_events(self):
        if not self.events:
            return True
        event = self.events[0]
        if isinstance(event, DocumentStartEvent):
            return self.need_events(1)
        elif isinstance(event, SequenceStartEvent):
            return self.need_events(2)
        elif isinstance(event, MappingStartEvent):
            return self.need_events(3)
        else:
            return False

    def need_events(self, count):
        level = 0
        for event in self.events[1:]:
            if isinstance(event, (DocumentStartEvent, CollectionStartEvent)):
                level += 1
            elif isinstance(event, (DocumentEndEvent, CollectionEndEvent)):
                level -= 1
            elif isinstance(event, StreamEndEvent):
                level = -1
            if level < 0:
                return False
        return (len(self.events) < count+1)

    def increase_indent(self, flow=False, indentless=False):
        self.indents.append(self.indent)
        if self.indent is None:
            if flow:
                self.indent = self.best_indent
            else:
                self.indent = 0
        elif not indentless:
            self.indent += self.best_indent

    # States.

    # Stream handlers.

    def expect_stream_start(self):
        if isinstance(self.event, StreamStartEvent):
            if self.event.encoding and not hasattr(self.stream, 'encoding'):
                self.encoding = self.event.encoding
            self.write_stream_start()
            self.state = self.expect_first_document_start
        else:
            raise EmitterError("expected StreamStartEvent, but got %s"
                    % self.event)

    def expect_nothing(self):
        raise EmitterError("expected nothing, but got %s" % self.event)

    # Document handlers.

    def expect_first_document_start(self):
        return self.expect_document_start(first=True)

    def expect_document_start(self, first=False):
        if isinstance(self.event, DocumentStartEvent):
            if (self.event.version or self.event.tags) and self.open_ended:
                self.write_indicator('...', True)
                self.write_indent()
            if self.event.version:
                version_text = self.prepare_version(self.event.version)
                self.write_version_directive(version_text)
            self.tag_prefixes = self.DEFAULT_TAG_PREFIXES.copy()
            if self.event.tags:
                handles = sorted(self.event.tags.keys())
                for handle in handles:
                    prefix = self.event.tags[handle]
                    self.tag_prefixes[prefix] = handle
                    handle_text = self.prepare_tag_handle(handle)
                    prefix_text = self.prepare_tag_prefix(prefix)
                    self.write_tag_directive(handle_text, prefix_text)
            implicit = (first and not self.event.explicit and not self.canonical
                    and not self.event.version and not self.event.tags
                    and not self.check_empty_document())
            if not implicit:
                self.write_indent()
                self.write_indicator('---', True)
                if self.canonical:
                    self.write_indent()
            self.state = self.expect_document_root
        elif isinstance(self.event, StreamEndEvent):
            if self.open_ended:
                self.write_indicator('...', True)
                self.write_indent()
            self.write_stream_end()
            self.state = self.expect_nothing
        else:
            raise EmitterError("expected DocumentStartEvent, but got %s"
                    % self.event)

    def expect_document_end(self):
        if isinstance(self.event, DocumentEndEvent):
            self.write_indent()
            if self.event.explicit:
                self.write_indicator('...', True)
                self.write_indent()
            self.flush_stream()
            self.state = self.expect_document_start
        else:
            raise EmitterError("expected DocumentEndEvent, but got %s"
                    % self.event)

    def expect_document_root(self):
        self.states.append(self.expect_document_end)
        self.expect_node(root=True)

    # Node handlers.

    def expect_node(self, root=False, sequence=False, mapping=False,
            simple_key=False):
        self.root_context = root
        self.sequence_context = sequence
        self.mapping_context = mapping
        self.simple_key_context = simple_key
        if isinstance(self.event, AliasEvent):
            self.expect_alias()
        elif isinstance(self.event, (ScalarEvent, CollectionStartEvent)):
            self.process_anchor('&')
            self.process_tag()
            if isinstance(self.event, ScalarEvent):
                self.expect_scalar()
            elif isinstance(self.event, SequenceStartEvent):
                if self.flow_level or self.canonical or self.event.flow_style   \
                        or self.check_empty_sequence():
                    self.expect_flow_sequence()
                else:
                    self.expect_block_sequence()
            elif isinstance(self.event, MappingStartEvent):
                if self.flow_level or self.canonical or self.event.flow_style   \
                        or self.check_empty_mapping():
                    self.expect_flow_mapping()
                else:
                    self.expect_block_mapping()
        else:
            raise EmitterError("expected NodeEvent, but got %s" % self.event)

    def expect_alias(self):
        if self.event.anchor is None:
            raise EmitterError("anchor is not specified for alias")
        self.process_anchor('*')
        self.state = self.states.pop()

    def expect_scalar(self):
        self.increase_indent(flow=True)
        self.process_scalar()
        self.indent = self.indents.pop()
        self.state = self.states.pop()

    # Flow sequence handlers.

    def expect_flow_sequence(self):
        self.write_indicator('[', True, whitespace=True)
        self.flow_level += 1
        self.increase_indent(flow=True)
        self.state = self.expect_first_flow_sequence_item

    def expect_first_flow_sequence_item(self):
        if isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            self.write_indicator(']', False)
            self.state = self.states.pop()
        else:
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            self.states.append(self.expect_flow_sequence_item)
            self.expect_node(sequence=True)

    def expect_flow_sequence_item(self):
        if isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            if self.canonical:
                self.write_indicator(',', False)
                self.write_indent()
            self.write_indicator(']', False)
            self.state = self.states.pop()
        else:
            self.write_indicator(',', False)
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            self.states.append(self.expect_flow_sequence_item)
            self.expect_node(sequence=True)

    # Flow mapping handlers.

    def expect_flow_mapping(self):
        self.write_indicator('{', True, whitespace=True)
        self.flow_level += 1
        self.increase_indent(flow=True)
        self.state = self.expect_first_flow_mapping_key

    def expect_first_flow_mapping_key(self):
        if isinstance(self.event, MappingEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            self.write_indicator('}', False)
            self.state = self.states.pop()
        else:
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            if not self.canonical and self.check_simple_key():
                self.states.append(self.expect_flow_mapping_simple_value)
                self.expect_node(mapping=True, simple_key=True)
            else:
                self.write_indicator('?', True)
                self.states.append(self.expect_flow_mapping_value)
                self.expect_node(mapping=True)

    def expect_flow_mapping_key(self):
        if isinstance(self.event, MappingEndEvent):
            self.indent = self.indents.pop()
            self.flow_level -= 1
            if self.canonical:
                self.write_indicator(',', False)
                self.write_indent()
            self.write_indicator('}', False)
            self.state = self.states.pop()
        else:
            self.write_indicator(',', False)
            if self.canonical or self.column > self.best_width:
                self.write_indent()
            if not self.canonical and self.check_simple_key():
                self.states.append(self.expect_flow_mapping_simple_value)
                self.expect_node(mapping=True, simple_key=True)
            else:
                self.write_indicator('?', True)
                self.states.append(self.expect_flow_mapping_value)
                self.expect_node(mapping=True)

    def expect_flow_mapping_simple_value(self):
        self.write_indicator(':', False)
        self.states.append(self.expect_flow_mapping_key)
        self.expect_node(mapping=True)

    def expect_flow_mapping_value(self):
        if self.canonical or self.column > self.best_width:
            self.write_indent()
        self.write_indicator(':', True)
        self.states.append(self.expect_flow_mapping_key)
        self.expect_node(mapping=True)

    # Block sequence handlers.

    def expect_block_sequence(self):
        indentless = (self.mapping_context and not self.indention)
        self.increase_indent(flow=False, indentless=indentless)
        self.state = self.expect_first_block_sequence_item

    def expect_first_block_sequence_item(self):
        return self.expect_block_sequence_item(first=True)

    def expect_block_sequence_item(self, first=False):
        if not first and isinstance(self.event, SequenceEndEvent):
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        else:
            self.write_indent()
            self.write_indicator('-', True, indention=True)
            self.states.append(self.expect_block_sequence_item)
            self.expect_node(sequence=True)

    # Block mapping handlers.

    def expect_block_mapping(self):
        self.increase_indent(flow=False)
        self.state = self.expect_first_block_mapping_key

    def expect_first_block_mapping_key(self):
        return self.expect_block_mapping_key(first=True)

    def expect_block_mapping_key(self, first=False):
        if not first and isinstance(self.event, MappingEndEvent):
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        else:
            self.write_indent()
            if self.check_simple_key():
                self.states.append(self.expect_block_mapping_simple_value)
                self.expect_node(mapping=True, simple_key=True)
            else:
                self.write_indicator('?', True, indention=True)
                self.states.append(self.expect_block_mapping_value)
                self.expect_node(mapping=True)

    def expect_block_mapping_simple_value(self):
        self.write_indicator(':', False)
        self.states.append(self.expect_block_mapping_key)
        self.expect_node(mapping=True)

    def expect_block_mapping_value(self):
        self.write_indent()
        self.write_indicator(':', True, indention=True)
        self.states.append(self.expect_block_mapping_key)
        self.expect_node(mapping=True)

    # Checkers.

    def check_empty_sequence(self):
        return (isinstance(self.event, SequenceStartEvent) and self.events
                and isinstance(self.events[0], SequenceEndEvent))

    def check_empty_mapping(self):
        return (isinstance(self.event, MappingStartEvent) and self.events
                and isinstance(self.events[0], MappingEndEvent))

    def check_empty_document(self):
        if not isinstance(self.event, DocumentStartEvent) or not self.events:
            return False
        event = self.events[0]
        return (isinstance(event, ScalarEvent) and event.anchor is None
                and event.tag is None and event.implicit and event.value == '')

    def check_simple_key(self):
        length = 0
        if isinstance(self.event, NodeEvent) and self.event.anchor is not None:
            if self.prepared_anchor is None:
                self.prepared_anchor = self.prepare_anchor(self.event.anchor)
            length += len(self.prepared_anchor)
        if isinstance(self.event, (ScalarEvent, CollectionStartEvent))  \
                and self.event.tag is not None:
            if self.prepared_tag is None:
                self.prepared_tag = self.prepare_tag(self.event.tag)
            length += len(self.prepared_tag)
        if isinstance(self.event, ScalarEvent):
            if self.analysis is None:
                self.analysis = self.analyze_scalar(self.event.value)
            length += len(self.analysis.scalar)
        return (length < 128 and (isinstance(self.event, AliasEvent)
            or (isinstance(self.event, ScalarEvent)
                    and not self.analysis.empty and not self.analysis.multiline)
            or self.check_empty_sequence() or self.check_empty_mapping()))

    # Anchor, Tag, and Scalar processors.

    def process_anchor(self, indicator):
        if self.event.anchor is None:
            self.prepared_anchor = None
            return
        if self.prepared_anchor is None:
            self.prepared_anchor = self.prepare_anchor(self.event.anchor)
        if self.prepared_anchor:
            self.write_indicator(indicator+self.prepared_anchor, True)
        self.prepared_anchor = None

    def process_tag(self):
        tag = self.event.tag
        if isinstance(self.event, ScalarEvent):
            if self.style is None:
                self.style = self.choose_scalar_style()
            if ((not self.canonical or tag is None) and
                ((self.style == '' and self.event.implicit[0])
                        or (self.style != '' and self.event.implicit[1]))):
                self.prepared_tag = None
                return
            if self.event.implicit[0] and tag is None:
                tag = '!'
                self.prepared_tag = None
        else:
            if (not self.canonical or tag is None) and self.event.implicit:
                self.prepared_tag = None
                return
        if tag is None:
            raise EmitterError("tag is not specified")
        if self.prepared_tag is None:
            self.prepared_tag = self.prepare_tag(tag)
        if self.prepared_tag:
            self.write_indicator(self.prepared_tag, True)
        self.prepared_tag = None

    def choose_scalar_style(self):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)
        if self.event.style == '"' or self.canonical:
            return '"'
        if not self.event.style and self.event.implicit[0]:
            if (not (self.simple_key_context and
                    (self.analysis.empty or self.analysis.multiline))
                and (self.flow_level and self.analysis.allow_flow_plain
                    or (not self.flow_level and self.analysis.allow_block_plain))):
                return ''
        if self.event.style and self.event.style in '|>':
            if (not self.flow_level and not self.simple_key_context
                    and self.analysis.allow_block):
                return self.event.style
        if not self.event.style or self.event.style == '\'':
            if (self.analysis.allow_single_quoted and
                    not (self.simple_key_context and self.analysis.multiline)):
                return '\''
        return '"'

    def process_scalar(self):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)
        if self.style is None:
            self.style = self.choose_scalar_style()
        split = (not self.simple_key_context)
        #if self.analysis.multiline and split    \
        #        and (not self.style or self.style in '\'\"'):
        #    self.write_indent()
        if self.style == '"':
            self.write_double_quoted(self.analysis.scalar, split)
        elif self.style == '\'':
            self.write_single_quoted(self.analysis.scalar, split)
        elif self.style == '>':
            self.write_folded(self.analysis.scalar)
        elif self.style == '|':
            self.write_literal(self.analysis.scalar)
        else:
            self.write_plain(self.analysis.scalar, split)
        self.analysis = None
        self.style = None

    # Analyzers.

    def prepare_version(self, version):
        major, minor = version
        if major != 1:
            raise EmitterError("unsupported YAML version: %d.%d" % (major, minor))
        return '%d.%d' % (major, minor)

    def prepare_tag_handle(self, handle):
        if not handle:
            raise EmitterError("tag handle must not be empty")
        if handle[0] != '!' or handle[-1] != '!':
            raise EmitterError("tag handle must start and end with '!': %r" % handle)
        for ch in handle[1:-1]:
            if not ('0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'    \
                    or ch in '-_'):
                raise EmitterError("invalid character %r in the tag handle: %r"
                        % (ch, handle))
        return handle

    def prepare_tag_prefix(self, prefix):
        if not prefix:
            raise EmitterError("tag prefix must not be empty")
        chunks = []
        start = end = 0
        if prefix[0] == '!':
            end = 1
        while end < len(prefix):
            ch = prefix[end]
            if '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' \
                    or ch in '-;/?!:@&=+$,_.~*\'()[]':
                end += 1
            else:
                if start < end:
                    chunks.append(prefix[start:end])
                start = end = end+1
                data = ch.encode('utf-8')
                for ch in data:
                    chunks.append('%%%02X' % ord(ch))
        if start < end:
            chunks.append(prefix[start:end])
        return ''.join(chunks)

    def prepare_tag(self, tag):
        if not tag:
            raise EmitterError("tag must not be empty")
        if tag == '!':
            return tag
        handle = None
        suffix = tag
        prefixes = sorted(self.tag_prefixes.keys())
        for prefix in prefixes:
            if tag.startswith(prefix)   \
                    and (prefix == '!' or len(prefix) < len(tag)):
                handle = self.tag_prefixes[prefix]
                suffix = tag[len(prefix):]
        chunks = []
        start = end = 0
        while end < len(suffix):
            ch = suffix[end]
            if '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' \
                    or ch in '-;/?:@&=+$,_.~*\'()[]'   \
                    or (ch == '!' and handle != '!'):
                end += 1
            else:
                if start < end:
                    chunks.append(suffix[start:end])
                start = end = end+1
                data = ch.encode('utf-8')
                for ch in data:
                    chunks.append('%%%02X' % ch)
        if start < end:
            chunks.append(suffix[start:end])
        suffix_text = ''.join(chunks)
        if handle:
            return '%s%s' % (handle, suffix_text)
        else:
            return '!<%s>' % suffix_text

    def prepare_anchor(self, anchor):
        if not anchor:
            raise EmitterError("anchor must not be empty")
        for ch in anchor:
            if not ('0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'    \
                    or ch in '-_'):
                raise EmitterError("invalid character %r in the anchor: %r"
                        % (ch, anchor))
        return anchor

    def analyze_scalar(self, scalar):

        # Empty scalar is a special case.
        if not scalar:
            return ScalarAnalysis(scalar=scalar, empty=True, multiline=False,
                    allow_flow_plain=False, allow_block_plain=True,
                    allow_single_quoted=True, allow_double_quoted=True,
                    allow_block=False)

        # Indicators and special characters.
        block_indicators = False
        flow_indicators = False
        line_breaks = False
        special_characters = False

        # Important whitespace combinations.
        leading_space = False
        leading_break = False
        trailing_space = False
        trailing_break = False
        break_space = False
        space_break = False

        # Check document indicators.
        if scalar.startswith('---') or scalar.startswith('...'):
            block_indicators = True
            flow_indicators = True

        # First character or preceded by a whitespace.
        preceded_by_whitespace = True

        # Last character or followed by a whitespace.
        followed_by_whitespace = (len(scalar) == 1 or
                scalar[1] in '\0 \t\r\n\x85\u2028\u2029')

        # The previous character is a space.
        previous_space = False

        # The previous character is a break.
        previous_break = False

        index = 0
        while index < len(scalar):
            ch = scalar[index]

            # Check for indicators.
            if index == 0:
                # Leading indicators are special characters.
                if ch in '#,[]{}&*!|>\'\"%@`':
                    flow_indicators = True
                    block_indicators = True
                if ch in '?:':
                    flow_indicators = True
                    if followed_by_whitespace:
                        block_indicators = True
                if ch == '-' and followed_by_whitespace:
                    flow_indicators = True
                    block_indicators = True
            else:
                # Some indicators cannot appear within a scalar as well.
                if ch in ',?[]{}':
                    flow_indicators = True
                if ch == ':':
                    flow_indicators = True
                    if followed_by_whitespace:
                        block_indicators = True
                if ch == '#' and preceded_by_whitespace:
                    flow_indicators = True
                    block_indicators = True

            # Check for line breaks, special, and unicode characters.
            if ch in '\n\x85\u2028\u2029':
                line_breaks = True
            if not (ch == '\n' or '\x20' <= ch <= '\x7E'):
                if (ch == '\x85' or '\xA0' <= ch <= '\uD7FF'
                        or '\uE000' <= ch <= '\uFFFD'
                        or '\U00010000' <= ch < '\U0010ffff') and ch != '\uFEFF':
                    unicode_characters = True
                    if not self.allow_unicode:
                        special_characters = True
                else:
                    special_characters = True

            # Detect important whitespace combinations.
            if ch == ' ':
                if index == 0:
                    leading_space = True
                if index == len(scalar)-1:
                    trailing_space = True
                if previous_break:
                    break_space = True
                previous_space = True
                previous_break = False
            elif ch in '\n\x85\u2028\u2029':
                if index == 0:
                    leading_break = True
                if index == len(scalar)-1:
                    trailing_break = True
                if previous_space:
                    space_break = True
                previous_space = False
                previous_break = True
            else:
                previous_space = False
                previous_break = False

            # Prepare for the next character.
            index += 1
            preceded_by_whitespace = (ch in '\0 \t\r\n\x85\u2028\u2029')
            followed_by_whitespace = (index+1 >= len(scalar) or
                    scalar[index+1] in '\0 \t\r\n\x85\u2028\u2029')

        # Let's decide what styles are allowed.
        allow_flow_plain = True
        allow_block_plain = True
        allow_single_quoted = True
        allow_double_quoted = True
        allow_block = True

        # Leading and trailing whitespaces are bad for plain scalars.
        if (leading_space or leading_break
                or trailing_space or trailing_break):
            allow_flow_plain = allow_block_plain = False

        # We do not permit trailing spaces for block scalars.
        if trailing_space:
            allow_block = False

        # Spaces at the beginning of a new line are only acceptable for block
        # scalars.
        if break_space:
            allow_flow_plain = allow_block_plain = allow_single_quoted = False

        # Spaces followed by breaks, as well as special character are only
        # allowed for double quoted scalars.
        if space_break or special_characters:
            allow_flow_plain = allow_block_plain =  \
            allow_single_quoted = allow_block = False

        # Although the plain scalar writer supports breaks, we never emit
        # multiline plain scalars.
        if line_breaks:
            allow_flow_plain = allow_block_plain = False

        # Flow indicators are forbidden for flow plain scalars.
        if flow_indicators:
            allow_flow_plain = False

        # Block indicators are forbidden for block plain scalars.
        if block_indicators:
            allow_block_plain = False

        return ScalarAnalysis(scalar=scalar,
                empty=False, multiline=line_breaks,
                allow_flow_plain=allow_flow_plain,
                allow_block_plain=allow_block_plain,
                allow_single_quoted=allow_single_quoted,
                allow_double_quoted=allow_double_quoted,
                allow_block=allow_block)

    # Writers.

    def flush_stream(self):
        if hasattr(self.stream, 'flush'):
            self.stream.flush()

    def write_stream_start(self):
        # Write BOM if needed.
        if self.encoding and self.encoding.startswith('utf-16'):
            self.stream.write('\uFEFF'.encode(self.encoding))

    def write_stream_end(self):
        self.flush_stream()

    def write_indicator(self, indicator, need_whitespace,
            whitespace=False, indention=False):
        if self.whitespace or not need_whitespace:
            data = indicator
        else:
            data = ' '+indicator
        self.whitespace = whitespace
        self.indention = self.indention and indention
        self.column += len(data)
        self.open_ended = False
        if self.encoding:
            data = data.encode(self.encoding)
        self.stream.write(data)

    def write_indent(self):
        indent = self.indent or 0
        if not self.indention or self.column > indent   \
                or (self.column == indent and not self.whitespace):
            self.write_line_break()
        if self.column < indent:
            self.whitespace = True
            data = ' '*(indent-self.column)
            self.column = indent
            if self.encoding:
                data = data.encode(self.encoding)
            self.stream.write(data)

    def write_line_break(self, data=None):
        if data is None:
            data = self.best_line_break
        self.whitespace = True
        self.indention = True
        self.line += 1
        self.column = 0
        if self.encoding:
            data = data.encode(self.encoding)
        self.stream.write(data)

    def write_version_directive(self, version_text):
        data = '%%YAML %s' % version_text
        if self.encoding:
            data = data.encode(self.encoding)
        self.stream.write(data)
        self.write_line_break()

    def write_tag_directive(self, handle_text, prefix_text):
        data = '%%TAG %s %s' % (handle_text, prefix_text)
        if self.encoding:
            data = data.encode(self.encoding)
        self.stream.write(data)
        self.write_line_break()

    # Scalar streams.

    def write_single_quoted(self, text, split=True):
        self.write_indicator('\'', True)
        spaces = False
        breaks = False
        start = end = 0
        while end <= len(text):
            ch = None
            if end < len(text):
                ch = text[end]
            if spaces:
                if ch is None or ch != ' ':
                    if start+1 == end and self.column > self.best_width and split   \
                            and start != 0 and end != len(text):
                        self.write_indent()
                    else:
                        data = text[start:end]
                        self.column += len(data)
                        if self.encoding:
                            data = data.encode(self.encoding)
                        self.stream.write(data)
                    start = end
            elif breaks:
                if ch is None or ch not in '\n\x85\u2028\u2029':
                    if text[start] == '\n':
                        self.write_line_break()
                    for br in text[start:end]:
                        if br == '\n':
                            self.write_line_break()
                        else:
                            self.write_line_break(br)
                    self.write_indent()
                    start = end
            else:
                if ch is None or ch in ' \n\x85\u2028\u2029' or ch == '\'':
                    if start < end:
                        data = text[start:end]
                        self.column += len(data)
                        if self.encoding:
                            data = data.encode(self.encoding)
                        self.stream.write(data)
                        start = end
            if ch == '\'':
                data = '\'\''
                self.column += 2
                if self.encoding:
                    data = data.encode(self.encoding)
                self.stream.write(data)
                start = end + 1
            if ch is not None:
                spaces = (ch == ' ')
                breaks = (ch in '\n\x85\u2028\u2029')
            end += 1
        self.write_indicator('\'', False)

    ESCAPE_REPLACEMENTS = {
        '\0':       '0',
        '\x07':     'a',
        '\x08':     'b',
        '\x09':     't',
        '\x0A':     'n',
        '\x0B':     'v',
        '\x0C':     'f',
        '\x0D':     'r',
        '\x1B':     'e',
        '\"':       '\"',
        '\\':       '\\',
        '\x85':     'N',
        '\xA0':     '_',
        '\u2028':   'L',
        '\u2029':   'P',
    }

    def write_double_quoted(self, text, split=True):
        self.write_indicator('"', True)
        start = end = 0
        while end <= len(text):
            ch = None
            if end < len(text):
                ch = text[end]
            if ch is None or ch in '"\\\x85\u2028\u2029\uFEFF' \
                    or not ('\x20' <= ch <= '\x7E'
                        or (self.allow_unicode
                            and ('\xA0' <= ch <= '\uD7FF'
                                or '\uE000' <= ch <= '\uFFFD'))):
                if start < end:
                    data = text[start:end]
                    self.column += len(data)
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                    start = end
                if ch is not None:
                    if ch in self.ESCAPE_REPLACEMENTS:
                        data = '\\'+self.ESCAPE_REPLACEMENTS[ch]
                    elif ch <= '\xFF':
                        data = '\\x%02X' % ord(ch)
                    elif ch <= '\uFFFF':
                        data = '\\u%04X' % ord(ch)
                    else:
                        data = '\\U%08X' % ord(ch)
                    self.column += len(data)
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                    start = end+1
            if 0 < end < len(text)-1 and (ch == ' ' or start >= end)    \
                    and self.column+(end-start) > self.best_width and split:
                data = text[start:end]+'\\'
                if start < end:
                    start = end
                self.column += len(data)
                if self.encoding:
                    data = data.encode(self.encoding)
                self.stream.write(data)
                self.write_indent()
                self.whitespace = False
                self.indention = False
                if text[start] == ' ':
                    data = '\\'
                    self.column += len(data)
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
            end += 1
        self.write_indicator('"', False)

    def determine_block_hints(self, text):
        hints = ''
        if text:
            if text[0] in ' \n\x85\u2028\u2029':
                hints += str(self.best_indent)
            if text[-1] not in '\n\x85\u2028\u2029':
                hints += '-'
            elif len(text) == 1 or text[-2] in '\n\x85\u2028\u2029':
                hints += '+'
        return hints

    def write_folded(self, text):
        hints = self.determine_block_hints(text)
        self.write_indicator('>'+hints, True)
        if hints[-1:] == '+':
            self.open_ended = True
        self.write_line_break()
        leading_space = True
        spaces = False
        breaks = True
        start = end = 0
        while end <= len(text):
            ch = None
            if end < len(text):
                ch = text[end]
            if breaks:
                if ch is None or ch not in '\n\x85\u2028\u2029':
                    if not leading_space and ch is not None and ch != ' '   \
                            and text[start] == '\n':
                        self.write_line_break()
                    leading_space = (ch == ' ')
                    for br in text[start:end]:
                        if br == '\n':
                            self.write_line_break()
                        else:
                            self.write_line_break(br)
                    if ch is not None:
                        self.write_indent()
                    start = end
            elif spaces:
                if ch != ' ':
                    if start+1 == end and self.column > self.best_width:
                        self.write_indent()
                    else:
                        data = text[start:end]
                        self.column += len(data)
                        if self.encoding:
                            data = data.encode(self.encoding)
                        self.stream.write(data)
                    start = end
            else:
                if ch is None or ch in ' \n\x85\u2028\u2029':
                    data = text[start:end]
                    self.column += len(data)
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                    if ch is None:
                        self.write_line_break()
                    start = end
            if ch is not None:
                breaks = (ch in '\n\x85\u2028\u2029')
                spaces = (ch == ' ')
            end += 1

    def write_literal(self, text):
        hints = self.determine_block_hints(text)
        self.write_indicator('|'+hints, True)
        if hints[-1:] == '+':
            self.open_ended = True
        self.write_line_break()
        breaks = True
        start = end = 0
        while end <= len(text):
            ch = None
            if end < len(text):
                ch = text[end]
            if breaks:
                if ch is None or ch not in '\n\x85\u2028\u2029':
                    for br in text[start:end]:
                        if br == '\n':
                            self.write_line_break()
                        else:
                            self.write_line_break(br)
                    if ch is not None:
                        self.write_indent()
                    start = end
            else:
                if ch is None or ch in '\n\x85\u2028\u2029':
                    data = text[start:end]
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                    if ch is None:
                        self.write_line_break()
                    start = end
            if ch is not None:
                breaks = (ch in '\n\x85\u2028\u2029')
            end += 1

    def write_plain(self, text, split=True):
        if self.root_context:
            self.open_ended = True
        if not text:
            return
        if not self.whitespace:
            data = ' '
            self.column += len(data)
            if self.encoding:
                data = data.encode(self.encoding)
            self.stream.write(data)
        self.whitespace = False
        self.indention = False
        spaces = False
        breaks = False
        start = end = 0
        while end <= len(text):
            ch = None
            if end < len(text):
                ch = text[end]
            if spaces:
                if ch != ' ':
                    if start+1 == end and self.column > self.best_width and split:
                        self.write_indent()
                        self.whitespace = False
                        self.indention = False
                    else:
                        data = text[start:end]
                        self.column += len(data)
                        if self.encoding:
                            data = data.encode(self.encoding)
                        self.stream.write(data)
                    start = end
            elif breaks:
                if ch not in '\n\x85\u2028\u2029':
                    if text[start] == '\n':
                        self.write_line_break()
                    for br in text[start:end]:
                        if br == '\n':
                            self.write_line_break()
                        else:
                            self.write_line_break(br)
                    self.write_indent()
                    self.whitespace = False
                    self.indention = False
                    start = end
            else:
                if ch is None or ch in ' \n\x85\u2028\u2029':
                    data = text[start:end]
                    self.column += len(data)
                    if self.encoding:
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                    start = end
            if ch is not None:
                spaces = (ch == ' ')
                breaks = (ch in '\n\x85\u2028\u2029')
            end += 1
<|endoftext|>
# The following YAML grammar is LL(1) and is parsed by a recursive descent
# parser.
#
# stream            ::= STREAM-START implicit_document? explicit_document* STREAM-END
# implicit_document ::= block_node DOCUMENT-END*
# explicit_document ::= DIRECTIVE* DOCUMENT-START block_node? DOCUMENT-END*
# block_node_or_indentless_sequence ::=
#                       ALIAS
#                       | properties (block_content | indentless_block_sequence)?
#                       | block_content
#                       | indentless_block_sequence
# block_node        ::= ALIAS
#                       | properties block_content?
#                       | block_content
# flow_node         ::= ALIAS
#                       | properties flow_content?
#                       | flow_content
# properties        ::= TAG ANCHOR? | ANCHOR TAG?
# block_content     ::= block_collection | flow_collection | SCALAR
# flow_content      ::= flow_collection | SCALAR
# block_collection  ::= block_sequence | block_mapping
# flow_collection   ::= flow_sequence | flow_mapping
# block_sequence    ::= BLOCK-SEQUENCE-START (BLOCK-ENTRY block_node?)* BLOCK-END
# indentless_sequence   ::= (BLOCK-ENTRY block_node?)+
# block_mapping     ::= BLOCK-MAPPING_START
#                       ((KEY block_node_or_indentless_sequence?)?
#                       (VALUE block_node_or_indentless_sequence?)?)*
#                       BLOCK-END
# flow_sequence     ::= FLOW-SEQUENCE-START
#                       (flow_sequence_entry FLOW-ENTRY)*
#                       flow_sequence_entry?
#                       FLOW-SEQUENCE-END
# flow_sequence_entry   ::= flow_node | KEY flow_node? (VALUE flow_node?)?
# flow_mapping      ::= FLOW-MAPPING-START
#                       (flow_mapping_entry FLOW-ENTRY)*
#                       flow_mapping_entry?
#                       FLOW-MAPPING-END
# flow_mapping_entry    ::= flow_node | KEY flow_node? (VALUE flow_node?)?
#
# FIRST sets:
#
# stream: { STREAM-START }
# explicit_document: { DIRECTIVE DOCUMENT-START }
# implicit_document: FIRST(block_node)
# block_node: { ALIAS TAG ANCHOR SCALAR BLOCK-SEQUENCE-START BLOCK-MAPPING-START FLOW-SEQUENCE-START FLOW-MAPPING-START }
# flow_node: { ALIAS ANCHOR TAG SCALAR FLOW-SEQUENCE-START FLOW-MAPPING-START }
# block_content: { BLOCK-SEQUENCE-START BLOCK-MAPPING-START FLOW-SEQUENCE-START FLOW-MAPPING-START SCALAR }
# flow_content: { FLOW-SEQUENCE-START FLOW-MAPPING-START SCALAR }
# block_collection: { BLOCK-SEQUENCE-START BLOCK-MAPPING-START }
# flow_collection: { FLOW-SEQUENCE-START FLOW-MAPPING-START }
# block_sequence: { BLOCK-SEQUENCE-START }
# block_mapping: { BLOCK-MAPPING-START }
# block_node_or_indentless_sequence: { ALIAS ANCHOR TAG SCALAR BLOCK-SEQUENCE-START BLOCK-MAPPING-START FLOW-SEQUENCE-START FLOW-MAPPING-START BLOCK-ENTRY }
# indentless_sequence: { ENTRY }
# flow_collection: { FLOW-SEQUENCE-START FLOW-MAPPING-START }
# flow_sequence: { FLOW-SEQUENCE-START }
# flow_mapping: { FLOW-MAPPING-START }
# flow_sequence_entry: { ALIAS ANCHOR TAG SCALAR FLOW-SEQUENCE-START FLOW-MAPPING-START KEY }
# flow_mapping_entry: { ALIAS ANCHOR TAG SCALAR FLOW-SEQUENCE-START FLOW-MAPPING-START KEY }

__all__ = ['Parser', 'ParserError']

from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *

class ParserError(MarkedYAMLError):
    pass

class Parser:
    # Since writing a recursive-descendant parser is a straightforward task, we
    # do not give many comments here.

    DEFAULT_TAGS = {
        '!':   '!',
        '!!':  'tag:yaml.org,2002:',
    }

    def __init__(self):
        self.current_event = None
        self.yaml_version = None
        self.tag_handles = {}
        self.states = []
        self.marks = []
        self.state = self.parse_stream_start

    def dispose(self):
        # Reset the state attributes (to clear self-references)
        self.states = []
        self.state = None

    def check_event(self, *choices):
        # Check the type of the next event.
        if self.current_event is None:
            if self.state:
                self.current_event = self.state()
        if self.current_event is not None:
            if not choices:
                return True
            for choice in choices:
                if isinstance(self.current_event, choice):
                    return True
        return False

    def peek_event(self):
        # Get the next event.
        if self.current_event is None:
            if self.state:
                self.current_event = self.state()
        return self.current_event

    def get_event(self):
        # Get the next event and proceed further.
        if self.current_event is None:
            if self.state:
                self.current_event = self.state()
        value = self.current_event
        self.current_event = None
        return value

    # stream    ::= STREAM-START implicit_document? explicit_document* STREAM-END
    # implicit_document ::= block_node DOCUMENT-END*
    # explicit_document ::= DIRECTIVE* DOCUMENT-START block_node? DOCUMENT-END*

    def parse_stream_start(self):

        # Parse the stream start.
        token = self.get_token()
        event = StreamStartEvent(token.start_mark, token.end_mark,
                encoding=token.encoding)

        # Prepare the next state.
        self.state = self.parse_implicit_document_start

        return event

    def parse_implicit_document_start(self):

        # Parse an implicit document.
        if not self.check_token(DirectiveToken, DocumentStartToken,
                StreamEndToken):
            self.tag_handles = self.DEFAULT_TAGS
            token = self.peek_token()
            start_mark = end_mark = token.start_mark
            event = DocumentStartEvent(start_mark, end_mark,
                    explicit=False)

            # Prepare the next state.
            self.states.append(self.parse_document_end)
            self.state = self.parse_block_node

            return event

        else:
            return self.parse_document_start()

    def parse_document_start(self):

        # Parse any extra document end indicators.
        while self.check_token(DocumentEndToken):
            self.get_token()

        # Parse an explicit document.
        if not self.check_token(StreamEndToken):
            token = self.peek_token()
            start_mark = token.start_mark
            version, tags = self.process_directives()
            if not self.check_token(DocumentStartToken):
                raise ParserError(None, None,
                        "expected '<document start>', but found %r"
                        % self.peek_token().id,
                        self.peek_token().start_mark)
            token = self.get_token()
            end_mark = token.end_mark
            event = DocumentStartEvent(start_mark, end_mark,
                    explicit=True, version=version, tags=tags)
            self.states.append(self.parse_document_end)
            self.state = self.parse_document_content
        else:
            # Parse the end of the stream.
            token = self.get_token()
            event = StreamEndEvent(token.start_mark, token.end_mark)
            assert not self.states
            assert not self.marks
            self.state = None
        return event

    def parse_document_end(self):

        # Parse the document end.
        token = self.peek_token()
        start_mark = end_mark = token.start_mark
        explicit = False
        if self.check_token(DocumentEndToken):
            token = self.get_token()
            end_mark = token.end_mark
            explicit = True
        event = DocumentEndEvent(start_mark, end_mark,
                explicit=explicit)

        # Prepare the next state.
        self.state = self.parse_document_start

        return event

    def parse_document_content(self):
        if self.check_token(DirectiveToken,
                DocumentStartToken, DocumentEndToken, StreamEndToken):
            event = self.process_empty_scalar(self.peek_token().start_mark)
            self.state = self.states.pop()
            return event
        else:
            return self.parse_block_node()

    def process_directives(self):
        self.yaml_version = None
        self.tag_handles = {}
        while self.check_token(DirectiveToken):
            token = self.get_token()
            if token.name == 'YAML':
                if self.yaml_version is not None:
                    raise ParserError(None, None,
                            "found duplicate YAML directive", token.start_mark)
                major, minor = token.value
                if major != 1:
                    raise ParserError(None, None,
                            "found incompatible YAML document (version 1.* is required)",
                            token.start_mark)
                self.yaml_version = token.value
            elif token.name == 'TAG':
                handle, prefix = token.value
                if handle in self.tag_handles:
                    raise ParserError(None, None,
                            "duplicate tag handle %r" % handle,
                            token.start_mark)
                self.tag_handles[handle] = prefix
        if self.tag_handles:
            value = self.yaml_version, self.tag_handles.copy()
        else:
            value = self.yaml_version, None
        for key in self.DEFAULT_TAGS:
            if key not in self.tag_handles:
                self.tag_handles[key] = self.DEFAULT_TAGS[key]
        return value

    # block_node_or_indentless_sequence ::= ALIAS
    #               | properties (block_content | indentless_block_sequence)?
    #               | block_content
    #               | indentless_block_sequence
    # block_node    ::= ALIAS
    #                   | properties block_content?
    #                   | block_content
    # flow_node     ::= ALIAS
    #                   | properties flow_content?
    #                   | flow_content
    # properties    ::= TAG ANCHOR? | ANCHOR TAG?
    # block_content     ::= block_collection | flow_collection | SCALAR
    # flow_content      ::= flow_collection | SCALAR
    # block_collection  ::= block_sequence | block_mapping
    # flow_collection   ::= flow_sequence | flow_mapping

    def parse_block_node(self):
        return self.parse_node(block=True)

    def parse_flow_node(self):
        return self.parse_node()

    def parse_block_node_or_indentless_sequence(self):
        return self.parse_node(block=True, indentless_sequence=True)

    def parse_node(self, block=False, indentless_sequence=False):
        if self.check_token(AliasToken):
            token = self.get_token()
            event = AliasEvent(token.value, token.start_mark, token.end_mark)
            self.state = self.states.pop()
        else:
            anchor = None
            tag = None
            start_mark = end_mark = tag_mark = None
            if self.check_token(AnchorToken):
                token = self.get_token()
                start_mark = token.start_mark
                end_mark = token.end_mark
                anchor = token.value
                if self.check_token(TagToken):
                    token = self.get_token()
                    tag_mark = token.start_mark
                    end_mark = token.end_mark
                    tag = token.value
            elif self.check_token(TagToken):
                token = self.get_token()
                start_mark = tag_mark = token.start_mark
                end_mark = token.end_mark
                tag = token.value
                if self.check_token(AnchorToken):
                    token = self.get_token()
                    end_mark = token.end_mark
                    anchor = token.value
            if tag is not None:
                handle, suffix = tag
                if handle is not None:
                    if handle not in self.tag_handles:
                        raise ParserError("while parsing a node", start_mark,
                                "found undefined tag handle %r" % handle,
                                tag_mark)
                    tag = self.tag_handles[handle]+suffix
                else:
                    tag = suffix
            #if tag == '!':
            #    raise ParserError("while parsing a node", start_mark,
            #            "found non-specific tag '!'", tag_mark,
            #            "Please check 'http://pyyaml.org/wiki/YAMLNonSpecificTag' and share your opinion.")
            if start_mark is None:
                start_mark = end_mark = self.peek_token().start_mark
            event = None
            implicit = (tag is None or tag == '!')
            if indentless_sequence and self.check_token(BlockEntryToken):
                end_mark = self.peek_token().end_mark
                event = SequenceStartEvent(anchor, tag, implicit,
                        start_mark, end_mark)
                self.state = self.parse_indentless_sequence_entry
            else:
                if self.check_token(ScalarToken):
                    token = self.get_token()
                    end_mark = token.end_mark
                    if (token.plain and tag is None) or tag == '!':
                        implicit = (True, False)
                    elif tag is None:
                        implicit = (False, True)
                    else:
                        implicit = (False, False)
                    event = ScalarEvent(anchor, tag, implicit, token.value,
                            start_mark, end_mark, style=token.style)
                    self.state = self.states.pop()
                elif self.check_token(FlowSequenceStartToken):
                    end_mark = self.peek_token().end_mark
                    event = SequenceStartEvent(anchor, tag, implicit,
                            start_mark, end_mark, flow_style=True)
                    self.state = self.parse_flow_sequence_first_entry
                elif self.check_token(FlowMappingStartToken):
                    end_mark = self.peek_token().end_mark
                    event = MappingStartEvent(anchor, tag, implicit,
                            start_mark, end_mark, flow_style=True)
                    self.state = self.parse_flow_mapping_first_key
                elif block and self.check_token(BlockSequenceStartToken):
                    end_mark = self.peek_token().start_mark
                    event = SequenceStartEvent(anchor, tag, implicit,
                            start_mark, end_mark, flow_style=False)
                    self.state = self.parse_block_sequence_first_entry
                elif block and self.check_token(BlockMappingStartToken):
                    end_mark = self.peek_token().start_mark
                    event = MappingStartEvent(anchor, tag, implicit,
                            start_mark, end_mark, flow_style=False)
                    self.state = self.parse_block_mapping_first_key
                elif anchor is not None or tag is not None:
                    # Empty scalars are allowed even if a tag or an anchor is
                    # specified.
                    event = ScalarEvent(anchor, tag, (implicit, False), '',
                            start_mark, end_mark)
                    self.state = self.states.pop()
                else:
                    if block:
                        node = 'block'
                    else:
                        node = 'flow'
                    token = self.peek_token()
                    raise ParserError("while parsing a %s node" % node, start_mark,
                            "expected the node content, but found %r" % token.id,
                            token.start_mark)
        return event

    # block_sequence ::= BLOCK-SEQUENCE-START (BLOCK-ENTRY block_node?)* BLOCK-END

    def parse_block_sequence_first_entry(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        return self.parse_block_sequence_entry()

    def parse_block_sequence_entry(self):
        if self.check_token(BlockEntryToken):
            token = self.get_token()
            if not self.check_token(BlockEntryToken, BlockEndToken):
                self.states.append(self.parse_block_sequence_entry)
                return self.parse_block_node()
            else:
                self.state = self.parse_block_sequence_entry
                return self.process_empty_scalar(token.end_mark)
        if not self.check_token(BlockEndToken):
            token = self.peek_token()
            raise ParserError("while parsing a block collection", self.marks[-1],
                    "expected <block end>, but found %r" % token.id, token.start_mark)
        token = self.get_token()
        event = SequenceEndEvent(token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event

    # indentless_sequence ::= (BLOCK-ENTRY block_node?)+

    def parse_indentless_sequence_entry(self):
        if self.check_token(BlockEntryToken):
            token = self.get_token()
            if not self.check_token(BlockEntryToken,
                    KeyToken, ValueToken, BlockEndToken):
                self.states.append(self.parse_indentless_sequence_entry)
                return self.parse_block_node()
            else:
                self.state = self.parse_indentless_sequence_entry
                return self.process_empty_scalar(token.end_mark)
        token = self.peek_token()
        event = SequenceEndEvent(token.start_mark, token.start_mark)
        self.state = self.states.pop()
        return event

    # block_mapping     ::= BLOCK-MAPPING_START
    #                       ((KEY block_node_or_indentless_sequence?)?
    #                       (VALUE block_node_or_indentless_sequence?)?)*
    #                       BLOCK-END

    def parse_block_mapping_first_key(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        return self.parse_block_mapping_key()

    def parse_block_mapping_key(self):
        if self.check_token(KeyToken):
            token = self.get_token()
            if not self.check_token(KeyToken, ValueToken, BlockEndToken):
                self.states.append(self.parse_block_mapping_value)
                return self.parse_block_node_or_indentless_sequence()
            else:
                self.state = self.parse_block_mapping_value
                return self.process_empty_scalar(token.end_mark)
        if not self.check_token(BlockEndToken):
            token = self.peek_token()
            raise ParserError("while parsing a block mapping", self.marks[-1],
                    "expected <block end>, but found %r" % token.id, token.start_mark)
        token = self.get_token()
        event = MappingEndEvent(token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event

    def parse_block_mapping_value(self):
        if self.check_token(ValueToken):
            token = self.get_token()
            if not self.check_token(KeyToken, ValueToken, BlockEndToken):
                self.states.append(self.parse_block_mapping_key)
                return self.parse_block_node_or_indentless_sequence()
            else:
                self.state = self.parse_block_mapping_key
                return self.process_empty_scalar(token.end_mark)
        else:
            self.state = self.parse_block_mapping_key
            token = self.peek_token()
            return self.process_empty_scalar(token.start_mark)

    # flow_sequence     ::= FLOW-SEQUENCE-START
    #                       (flow_sequence_entry FLOW-ENTRY)*
    #                       flow_sequence_entry?
    #                       FLOW-SEQUENCE-END
    # flow_sequence_entry   ::= flow_node | KEY flow_node? (VALUE flow_node?)?
    #
    # Note that while production rules for both flow_sequence_entry and
    # flow_mapping_entry are equal, their interpretations are different.
    # For `flow_sequence_entry`, the part `KEY flow_node? (VALUE flow_node?)?`
    # generate an inline mapping (set syntax).

    def parse_flow_sequence_first_entry(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        return self.parse_flow_sequence_entry(first=True)

    def parse_flow_sequence_entry(self, first=False):
        if not self.check_token(FlowSequenceEndToken):
            if not first:
                if self.check_token(FlowEntryToken):
                    self.get_token()
                else:
                    token = self.peek_token()
                    raise ParserError("while parsing a flow sequence", self.marks[-1],
                            "expected ',' or ']', but got %r" % token.id, token.start_mark)
            
            if self.check_token(KeyToken):
                token = self.peek_token()
                event = MappingStartEvent(None, None, True,
                        token.start_mark, token.end_mark,
                        flow_style=True)
                self.state = self.parse_flow_sequence_entry_mapping_key
                return event
            elif not self.check_token(FlowSequenceEndToken):
                self.states.append(self.parse_flow_sequence_entry)
                return self.parse_flow_node()
        token = self.get_token()
        event = SequenceEndEvent(token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event

    def parse_flow_sequence_entry_mapping_key(self):
        token = self.get_token()
        if not self.check_token(ValueToken,
                FlowEntryToken, FlowSequenceEndToken):
            self.states.append(self.parse_flow_sequence_entry_mapping_value)
            return self.parse_flow_node()
        else:
            self.state = self.parse_flow_sequence_entry_mapping_value
            return self.process_empty_scalar(token.end_mark)

    def parse_flow_sequence_entry_mapping_value(self):
        if self.check_token(ValueToken):
            token = self.get_token()
            if not self.check_token(FlowEntryToken, FlowSequenceEndToken):
                self.states.append(self.parse_flow_sequence_entry_mapping_end)
                return self.parse_flow_node()
            else:
                self.state = self.parse_flow_sequence_entry_mapping_end
                return self.process_empty_scalar(token.end_mark)
        else:
            self.state = self.parse_flow_sequence_entry_mapping_end
            token = self.peek_token()
            return self.process_empty_scalar(token.start_mark)

    def parse_flow_sequence_entry_mapping_end(self):
        self.state = self.parse_flow_sequence_entry
        token = self.peek_token()
        return MappingEndEvent(token.start_mark, token.start_mark)

    # flow_mapping  ::= FLOW-MAPPING-START
    #                   (flow_mapping_entry FLOW-ENTRY)*
    #                   flow_mapping_entry?
    #                   FLOW-MAPPING-END
    # flow_mapping_entry    ::= flow_node | KEY flow_node? (VALUE flow_node?)?

    def parse_flow_mapping_first_key(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        return self.parse_flow_mapping_key(first=True)

    def parse_flow_mapping_key(self, first=False):
        if not self.check_token(FlowMappingEndToken):
            if not first:
                if self.check_token(FlowEntryToken):
                    self.get_token()
                else:
                    token = self.peek_token()
                    raise ParserError("while parsing a flow mapping", self.marks[-1],
                            "expected ',' or '}', but got %r" % token.id, token.start_mark)
            if self.check_token(KeyToken):
                token = self.get_token()
                if not self.check_token(ValueToken,
                        FlowEntryToken, FlowMappingEndToken):
                    self.states.append(self.parse_flow_mapping_value)
                    return self.parse_flow_node()
                else:
                    self.state = self.parse_flow_mapping_value
                    return self.process_empty_scalar(token.end_mark)
            elif not self.check_token(FlowMappingEndToken):
                self.states.append(self.parse_flow_mapping_empty_value)
                return self.parse_flow_node()
        token = self.get_token()
        event = MappingEndEvent(token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event

    def parse_flow_mapping_value(self):
        if self.check_token(ValueToken):
            token = self.get_token()
            if not self.check_token(FlowEntryToken, FlowMappingEndToken):
                self.states.append(self.parse_flow_mapping_key)
                return self.parse_flow_node()
            else:
                self.state = self.parse_flow_mapping_key
                return self.process_empty_scalar(token.end_mark)
        else:
            self.state = self.parse_flow_mapping_key
            token = self.peek_token()
            return self.process_empty_scalar(token.start_mark)

    def parse_flow_mapping_empty_value(self):
        self.state = self.parse_flow_mapping_key
        return self.process_empty_scalar(self.peek_token().start_mark)

    def process_empty_scalar(self, mark):
        return ScalarEvent(None, None, (True, False), '', mark, mark)

<|endoftext|>
class Token(object):
    def __init__(self, start_mark, end_mark):
        self.start_mark = start_mark
        self.end_mark = end_mark
    def __repr__(self):
        attributes = [key for key in self.__dict__
                if not key.endswith('_mark')]
        attributes.sort()
        arguments = ', '.join(['%s=%r' % (key, getattr(self, key))
                for key in attributes])
        return '%s(%s)' % (self.__class__.__name__, arguments)

#class BOMToken(Token):
#    id = '<byte order mark>'

class DirectiveToken(Token):
    id = '<directive>'
    def __init__(self, name, value, start_mark, end_mark):
        self.name = name
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark

class DocumentStartToken(Token):
    id = '<document start>'

class DocumentEndToken(Token):
    id = '<document end>'

class StreamStartToken(Token):
    id = '<stream start>'
    def __init__(self, start_mark=None, end_mark=None,
            encoding=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.encoding = encoding

class StreamEndToken(Token):
    id = '<stream end>'

class BlockSequenceStartToken(Token):
    id = '<block sequence start>'

class BlockMappingStartToken(Token):
    id = '<block mapping start>'

class BlockEndToken(Token):
    id = '<block end>'

class FlowSequenceStartToken(Token):
    id = '['

class FlowMappingStartToken(Token):
    id = '{'

class FlowSequenceEndToken(Token):
    id = ']'

class FlowMappingEndToken(Token):
    id = '}'

class KeyToken(Token):
    id = '?'

class ValueToken(Token):
    id = ':'

class BlockEntryToken(Token):
    id = '-'

class FlowEntryToken(Token):
    id = ','

class AliasToken(Token):
    id = '<alias>'
    def __init__(self, value, start_mark, end_mark):
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark

class AnchorToken(Token):
    id = '<anchor>'
    def __init__(self, value, start_mark, end_mark):
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark

class TagToken(Token):
    id = '<tag>'
    def __init__(self, value, start_mark, end_mark):
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark

class ScalarToken(Token):
    id = '<scalar>'
    def __init__(self, value, plain, start_mark, end_mark, style=None):
        self.value = value
        self.plain = plain
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.style = style

<|endoftext|>
__all__ = [
    'CBaseLoader', 'CSafeLoader', 'CFullLoader', 'CUnsafeLoader', 'CLoader',
    'CBaseDumper', 'CSafeDumper', 'CDumper'
]

from yaml._yaml import CParser, CEmitter

from .constructor import *

from .serializer import *
from .representer import *

from .resolver import *

class CBaseLoader(CParser, BaseConstructor, BaseResolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        BaseConstructor.__init__(self)
        BaseResolver.__init__(self)

class CSafeLoader(CParser, SafeConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        SafeConstructor.__init__(self)
        Resolver.__init__(self)

class CFullLoader(CParser, FullConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        FullConstructor.__init__(self)
        Resolver.__init__(self)

class CUnsafeLoader(CParser, UnsafeConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        UnsafeConstructor.__init__(self)
        Resolver.__init__(self)

class CLoader(CParser, Constructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        Constructor.__init__(self)
        Resolver.__init__(self)

class CBaseDumper(CEmitter, BaseRepresenter, BaseResolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        CEmitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width, encoding=encoding,
                allow_unicode=allow_unicode, line_break=line_break,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        Representer.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

class CSafeDumper(CEmitter, SafeRepresenter, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        CEmitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width, encoding=encoding,
                allow_unicode=allow_unicode, line_break=line_break,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        SafeRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

class CDumper(CEmitter, Serializer, Representer, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        CEmitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width, encoding=encoding,
                allow_unicode=allow_unicode, line_break=line_break,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        Representer.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

<|endoftext|>
"""
opcode module - potentially shared between dis and other modules which
operate on bytecodes (e.g. peephole optimizers).
"""

__all__ = ["cmp_op", "hasconst", "hasname", "hasjrel", "hasjabs",
           "haslocal", "hascompare", "hasfree", "opname", "opmap",
           "HAVE_ARGUMENT", "EXTENDED_ARG", "hasnargs"]

# It's a chicken-and-egg I'm afraid:
# We're imported before _opcode's made.
# With exception unheeded
# (stack_effect is not needed)
# Both our chickens and eggs are allayed.
#     --Larry Hastings, 2013/11/23

try:
    from _opcode import stack_effect
    __all__.append('stack_effect')
except ImportError:
    pass

cmp_op = ('<', '<=', '==', '!=', '>', '>=')

hasconst = []
hasname = []
hasjrel = []
hasjabs = []
haslocal = []
hascompare = []
hasfree = []
hasnargs = [] # unused

opmap = {}
opname = ['<%r>' % (op,) for op in range(256)]

def def_op(name, op):
    opname[op] = name
    opmap[name] = op

def name_op(name, op):
    def_op(name, op)
    hasname.append(op)

def jrel_op(name, op):
    def_op(name, op)
    hasjrel.append(op)

def jabs_op(name, op):
    def_op(name, op)
    hasjabs.append(op)

# Instruction opcodes for compiled code
# Blank lines correspond to available opcodes

def_op('POP_TOP', 1)
def_op('ROT_TWO', 2)
def_op('ROT_THREE', 3)
def_op('DUP_TOP', 4)
def_op('DUP_TOP_TWO', 5)
def_op('ROT_FOUR', 6)

def_op('NOP', 9)
def_op('UNARY_POSITIVE', 10)
def_op('UNARY_NEGATIVE', 11)
def_op('UNARY_NOT', 12)

def_op('UNARY_INVERT', 15)
def_op('BINARY_MATRIX_MULTIPLY', 16)
def_op('INPLACE_MATRIX_MULTIPLY', 17)

def_op('BINARY_POWER', 19)
def_op('BINARY_MULTIPLY', 20)

def_op('BINARY_MODULO', 22)
def_op('BINARY_ADD', 23)
def_op('BINARY_SUBTRACT', 24)
def_op('BINARY_SUBSCR', 25)
def_op('BINARY_FLOOR_DIVIDE', 26)
def_op('BINARY_TRUE_DIVIDE', 27)
def_op('INPLACE_FLOOR_DIVIDE', 28)
def_op('INPLACE_TRUE_DIVIDE', 29)
def_op('GET_LEN', 30)
def_op('MATCH_MAPPING', 31)
def_op('MATCH_SEQUENCE', 32)
def_op('MATCH_KEYS', 33)
def_op('COPY_DICT_WITHOUT_KEYS', 34)

def_op('WITH_EXCEPT_START', 49)
def_op('GET_AITER', 50)
def_op('GET_ANEXT', 51)
def_op('BEFORE_ASYNC_WITH', 52)

def_op('END_ASYNC_FOR', 54)
def_op('INPLACE_ADD', 55)
def_op('INPLACE_SUBTRACT', 56)
def_op('INPLACE_MULTIPLY', 57)

def_op('INPLACE_MODULO', 59)
def_op('STORE_SUBSCR', 60)
def_op('DELETE_SUBSCR', 61)
def_op('BINARY_LSHIFT', 62)
def_op('BINARY_RSHIFT', 63)
def_op('BINARY_AND', 64)
def_op('BINARY_XOR', 65)
def_op('BINARY_OR', 66)
def_op('INPLACE_POWER', 67)
def_op('GET_ITER', 68)
def_op('GET_YIELD_FROM_ITER', 69)
def_op('PRINT_EXPR', 70)
def_op('LOAD_BUILD_CLASS', 71)
def_op('YIELD_FROM', 72)
def_op('GET_AWAITABLE', 73)
def_op('LOAD_ASSERTION_ERROR', 74)
def_op('INPLACE_LSHIFT', 75)
def_op('INPLACE_RSHIFT', 76)
def_op('INPLACE_AND', 77)
def_op('INPLACE_XOR', 78)
def_op('INPLACE_OR', 79)

def_op('LIST_TO_TUPLE', 82)
def_op('RETURN_VALUE', 83)
def_op('IMPORT_STAR', 84)
def_op('SETUP_ANNOTATIONS', 85)
def_op('YIELD_VALUE', 86)
def_op('POP_BLOCK', 87)

def_op('POP_EXCEPT', 89)

HAVE_ARGUMENT = 90              # Opcodes from here have an argument:

name_op('STORE_NAME', 90)       # Index in name list
name_op('DELETE_NAME', 91)      # ""
def_op('UNPACK_SEQUENCE', 92)   # Number of tuple items
jrel_op('FOR_ITER', 93)
def_op('UNPACK_EX', 94)
name_op('STORE_ATTR', 95)       # Index in name list
name_op('DELETE_ATTR', 96)      # ""
name_op('STORE_GLOBAL', 97)     # ""
name_op('DELETE_GLOBAL', 98)    # ""
def_op('ROT_N', 99)
def_op('LOAD_CONST', 100)       # Index in const list
hasconst.append(100)
name_op('LOAD_NAME', 101)       # Index in name list
def_op('BUILD_TUPLE', 102)      # Number of tuple items
def_op('BUILD_LIST', 103)       # Number of list items
def_op('BUILD_SET', 104)        # Number of set items
def_op('BUILD_MAP', 105)        # Number of dict entries
name_op('LOAD_ATTR', 106)       # Index in name list
def_op('COMPARE_OP', 107)       # Comparison operator
hascompare.append(107)
name_op('IMPORT_NAME', 108)     # Index in name list
name_op('IMPORT_FROM', 109)     # Index in name list
jrel_op('JUMP_FORWARD', 110)    # Number of bytes to skip
jabs_op('JUMP_IF_FALSE_OR_POP', 111) # Target byte offset from beginning of code
jabs_op('JUMP_IF_TRUE_OR_POP', 112)  # ""
jabs_op('JUMP_ABSOLUTE', 113)        # ""
jabs_op('POP_JUMP_IF_FALSE', 114)    # ""
jabs_op('POP_JUMP_IF_TRUE', 115)     # ""
name_op('LOAD_GLOBAL', 116)     # Index in name list
def_op('IS_OP', 117)
def_op('CONTAINS_OP', 118)
def_op('RERAISE', 119)

jabs_op('JUMP_IF_NOT_EXC_MATCH', 121)
jrel_op('SETUP_FINALLY', 122)   # Distance to target address

def_op('LOAD_FAST', 124)        # Local variable number
haslocal.append(124)
def_op('STORE_FAST', 125)       # Local variable number
haslocal.append(125)
def_op('DELETE_FAST', 126)      # Local variable number
haslocal.append(126)

def_op('GEN_START', 129)        # Kind of generator/coroutine
def_op('RAISE_VARARGS', 130)    # Number of raise arguments (1, 2, or 3)
def_op('CALL_FUNCTION', 131)    # #args
def_op('MAKE_FUNCTION', 132)    # Flags
def_op('BUILD_SLICE', 133)      # Number of items

def_op('LOAD_CLOSURE', 135)
hasfree.append(135)
def_op('LOAD_DEREF', 136)
hasfree.append(136)
def_op('STORE_DEREF', 137)
hasfree.append(137)
def_op('DELETE_DEREF', 138)
hasfree.append(138)

def_op('CALL_FUNCTION_KW', 141)  # #args + #kwargs
def_op('CALL_FUNCTION_EX', 142)  # Flags
jrel_op('SETUP_WITH', 143)
def_op('EXTENDED_ARG', 144)
EXTENDED_ARG = 144
def_op('LIST_APPEND', 145)
def_op('SET_ADD', 146)
def_op('MAP_ADD', 147)
def_op('LOAD_CLASSDEREF', 148)
hasfree.append(148)

def_op('MATCH_CLASS', 152)

jrel_op('SETUP_ASYNC_WITH', 154)
def_op('FORMAT_VALUE', 155)
def_op('BUILD_CONST_KEY_MAP', 156)
def_op('BUILD_STRING', 157)

name_op('LOAD_METHOD', 160)
def_op('CALL_METHOD', 161)
def_op('LIST_EXTEND', 162)
def_op('SET_UPDATE', 163)
def_op('DICT_MERGE', 164)
def_op('DICT_UPDATE', 165)

del def_op, name_op, jrel_op, jabs_op
<|endoftext|>"""Read/write support for Maildir, mbox, MH, Babyl, and MMDF mailboxes."""

# Notes for authors of new mailbox subclasses:
#
# Remember to fsync() changes to disk before closing a modified file
# or returning from a flush() method.  See functions _sync_flush() and
# _sync_close().

import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
try:
    import fcntl
except ImportError:
    fcntl = None

__all__ = ['Mailbox', 'Maildir', 'mbox', 'MH', 'Babyl', 'MMDF',
           'Message', 'MaildirMessage', 'mboxMessage', 'MHMessage',
           'BabylMessage', 'MMDFMessage', 'Error', 'NoSuchMailboxError',
           'NotEmptyError', 'ExternalClashError', 'FormatError']

linesep = os.linesep.encode('ascii')

class Mailbox:
    """A group of messages in a particular place."""

    def __init__(self, path, factory=None, create=True):
        """Initialize a Mailbox instance."""
        self._path = os.path.abspath(os.path.expanduser(path))
        self._factory = factory

    def add(self, message):
        """Add message and return assigned key."""
        raise NotImplementedError('Method must be implemented by subclass')

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        raise NotImplementedError('Method must be implemented by subclass')

    def __delitem__(self, key):
        self.remove(key)

    def discard(self, key):
        """If the keyed message exists, remove it."""
        try:
            self.remove(key)
        except KeyError:
            pass

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        raise NotImplementedError('Method must be implemented by subclass')

    def get(self, key, default=None):
        """Return the keyed message, or default if it doesn't exist."""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __getitem__(self, key):
        """Return the keyed message; raise KeyError if it doesn't exist."""
        if not self._factory:
            return self.get_message(key)
        else:
            with contextlib.closing(self.get_file(key)) as file:
                return self._factory(file)

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        raise NotImplementedError('Method must be implemented by subclass')

    def get_string(self, key):
        """Return a string representation or raise a KeyError.

        Uses email.message.Message to create a 7bit clean string
        representation of the message."""
        return email.message_from_bytes(self.get_bytes(key)).as_string()

    def get_bytes(self, key):
        """Return a byte string representation or raise a KeyError."""
        raise NotImplementedError('Method must be implemented by subclass')

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        raise NotImplementedError('Method must be implemented by subclass')

    def iterkeys(self):
        """Return an iterator over keys."""
        raise NotImplementedError('Method must be implemented by subclass')

    def keys(self):
        """Return a list of keys."""
        return list(self.iterkeys())

    def itervalues(self):
        """Return an iterator over all messages."""
        for key in self.iterkeys():
            try:
                value = self[key]
            except KeyError:
                continue
            yield value

    def __iter__(self):
        return self.itervalues()

    def values(self):
        """Return a list of messages. Memory intensive."""
        return list(self.itervalues())

    def iteritems(self):
        """Return an iterator over (key, message) tuples."""
        for key in self.iterkeys():
            try:
                value = self[key]
            except KeyError:
                continue
            yield (key, value)

    def items(self):
        """Return a list of (key, message) tuples. Memory intensive."""
        return list(self.iteritems())

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        raise NotImplementedError('Method must be implemented by subclass')

    def __len__(self):
        """Return a count of messages in the mailbox."""
        raise NotImplementedError('Method must be implemented by subclass')

    def clear(self):
        """Delete all messages."""
        for key in self.keys():
            self.discard(key)

    def pop(self, key, default=None):
        """Delete the keyed message and return it, or default."""
        try:
            result = self[key]
        except KeyError:
            return default
        self.discard(key)
        return result

    def popitem(self):
        """Delete an arbitrary (key, message) pair and return it."""
        for key in self.iterkeys():
            return (key, self.pop(key))     # This is only run once.
        else:
            raise KeyError('No messages in mailbox')

    def update(self, arg=None):
        """Change the messages that correspond to certain keys."""
        if hasattr(arg, 'iteritems'):
            source = arg.iteritems()
        elif hasattr(arg, 'items'):
            source = arg.items()
        else:
            source = arg
        bad_key = False
        for key, message in source:
            try:
                self[key] = message
            except KeyError:
                bad_key = True
        if bad_key:
            raise KeyError('No message with key(s)')

    def flush(self):
        """Write any pending changes to the disk."""
        raise NotImplementedError('Method must be implemented by subclass')

    def lock(self):
        """Lock the mailbox."""
        raise NotImplementedError('Method must be implemented by subclass')

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        raise NotImplementedError('Method must be implemented by subclass')

    def close(self):
        """Flush and close the mailbox."""
        raise NotImplementedError('Method must be implemented by subclass')

    def _string_to_bytes(self, message):
        # If a message is not 7bit clean, we refuse to handle it since it
        # likely came from reading invalid messages in text mode, and that way
        # lies mojibake.
        try:
            return message.encode('ascii')
        except UnicodeError:
            raise ValueError("String input must be ASCII-only; "
                "use bytes or a Message instead")

    # Whether each message must end in a newline
    _append_newline = False

    def _dump_message(self, message, target, mangle_from_=False):
        # This assumes the target file is open in binary mode.
        """Dump message contents to target file."""
        if isinstance(message, email.message.Message):
            buffer = io.BytesIO()
            gen = email.generator.BytesGenerator(buffer, mangle_from_, 0)
            gen.flatten(message)
            buffer.seek(0)
            data = buffer.read()
            data = data.replace(b'\n', linesep)
            target.write(data)
            if self._append_newline and not data.endswith(linesep):
                # Make sure the message ends with a newline
                target.write(linesep)
        elif isinstance(message, (str, bytes, io.StringIO)):
            if isinstance(message, io.StringIO):
                warnings.warn("Use of StringIO input is deprecated, "
                    "use BytesIO instead", DeprecationWarning, 3)
                message = message.getvalue()
            if isinstance(message, str):
                message = self._string_to_bytes(message)
            if mangle_from_:
                message = message.replace(b'\nFrom ', b'\n>From ')
            message = message.replace(b'\n', linesep)
            target.write(message)
            if self._append_newline and not message.endswith(linesep):
                # Make sure the message ends with a newline
                target.write(linesep)
        elif hasattr(message, 'read'):
            if hasattr(message, 'buffer'):
                warnings.warn("Use of text mode files is deprecated, "
                    "use a binary mode file instead", DeprecationWarning, 3)
                message = message.buffer
            lastline = None
            while True:
                line = message.readline()
                # Universal newline support.
                if line.endswith(b'\r\n'):
                    line = line[:-2] + b'\n'
                elif line.endswith(b'\r'):
                    line = line[:-1] + b'\n'
                if not line:
                    break
                if mangle_from_ and line.startswith(b'From '):
                    line = b'>From ' + line[5:]
                line = line.replace(b'\n', linesep)
                target.write(line)
                lastline = line
            if self._append_newline and lastline and not lastline.endswith(linesep):
                # Make sure the message ends with a newline
                target.write(linesep)
        else:
            raise TypeError('Invalid message type: %s' % type(message))

    __class_getitem__ = classmethod(GenericAlias)


class Maildir(Mailbox):
    """A qmail-style Maildir mailbox."""

    colon = ':'

    def __init__(self, dirname, factory=None, create=True):
        """Initialize a Maildir instance."""
        Mailbox.__init__(self, dirname, factory, create)
        self._paths = {
            'tmp': os.path.join(self._path, 'tmp'),
            'new': os.path.join(self._path, 'new'),
            'cur': os.path.join(self._path, 'cur'),
            }
        if not os.path.exists(self._path):
            if create:
                os.mkdir(self._path, 0o700)
                for path in self._paths.values():
                    os.mkdir(path, 0o700)
            else:
                raise NoSuchMailboxError(self._path)
        self._toc = {}
        self._toc_mtimes = {'cur': 0, 'new': 0}
        self._last_read = 0         # Records last time we read cur/new
        self._skewfactor = 0.1      # Adjust if os/fs clocks are skewing

    def add(self, message):
        """Add message and return assigned key."""
        tmp_file = self._create_tmp()
        try:
            self._dump_message(message, tmp_file)
        except BaseException:
            tmp_file.close()
            os.remove(tmp_file.name)
            raise
        _sync_close(tmp_file)
        if isinstance(message, MaildirMessage):
            subdir = message.get_subdir()
            suffix = self.colon + message.get_info()
            if suffix == self.colon:
                suffix = ''
        else:
            subdir = 'new'
            suffix = ''
        uniq = os.path.basename(tmp_file.name).split(self.colon)[0]
        dest = os.path.join(self._path, subdir, uniq + suffix)
        if isinstance(message, MaildirMessage):
            os.utime(tmp_file.name,
                     (os.path.getatime(tmp_file.name), message.get_date()))
        # No file modification should be done after the file is moved to its
        # final position in order to prevent race conditions with changes
        # from other programs
        try:
            try:
                os.link(tmp_file.name, dest)
            except (AttributeError, PermissionError):
                os.rename(tmp_file.name, dest)
            else:
                os.remove(tmp_file.name)
        except OSError as e:
            os.remove(tmp_file.name)
            if e.errno == errno.EEXIST:
                raise ExternalClashError('Name clash with existing message: %s'
                                         % dest)
            else:
                raise
        return uniq

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        os.remove(os.path.join(self._path, self._lookup(key)))

    def discard(self, key):
        """If the keyed message exists, remove it."""
        # This overrides an inapplicable implementation in the superclass.
        try:
            self.remove(key)
        except (KeyError, FileNotFoundError):
            pass

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        old_subpath = self._lookup(key)
        temp_key = self.add(message)
        temp_subpath = self._lookup(temp_key)
        if isinstance(message, MaildirMessage):
            # temp's subdir and suffix were specified by message.
            dominant_subpath = temp_subpath
        else:
            # temp's subdir and suffix were defaults from add().
            dominant_subpath = old_subpath
        subdir = os.path.dirname(dominant_subpath)
        if self.colon in dominant_subpath:
            suffix = self.colon + dominant_subpath.split(self.colon)[-1]
        else:
            suffix = ''
        self.discard(key)
        tmp_path = os.path.join(self._path, temp_subpath)
        new_path = os.path.join(self._path, subdir, key + suffix)
        if isinstance(message, MaildirMessage):
            os.utime(tmp_path,
                     (os.path.getatime(tmp_path), message.get_date()))
        # No file modification should be done after the file is moved to its
        # final position in order to prevent race conditions with changes
        # from other programs
        os.rename(tmp_path, new_path)

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        subpath = self._lookup(key)
        with open(os.path.join(self._path, subpath), 'rb') as f:
            if self._factory:
                msg = self._factory(f)
            else:
                msg = MaildirMessage(f)
        subdir, name = os.path.split(subpath)
        msg.set_subdir(subdir)
        if self.colon in name:
            msg.set_info(name.split(self.colon)[-1])
        msg.set_date(os.path.getmtime(os.path.join(self._path, subpath)))
        return msg

    def get_bytes(self, key):
        """Return a bytes representation or raise a KeyError."""
        with open(os.path.join(self._path, self._lookup(key)), 'rb') as f:
            return f.read().replace(linesep, b'\n')

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        f = open(os.path.join(self._path, self._lookup(key)), 'rb')
        return _ProxyFile(f)

    def iterkeys(self):
        """Return an iterator over keys."""
        self._refresh()
        for key in self._toc:
            try:
                self._lookup(key)
            except KeyError:
                continue
            yield key

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        self._refresh()
        return key in self._toc

    def __len__(self):
        """Return a count of messages in the mailbox."""
        self._refresh()
        return len(self._toc)

    def flush(self):
        """Write any pending changes to disk."""
        # Maildir changes are always written immediately, so there's nothing
        # to do.
        pass

    def lock(self):
        """Lock the mailbox."""
        return

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        return

    def close(self):
        """Flush and close the mailbox."""
        return

    def list_folders(self):
        """Return a list of folder names."""
        result = []
        for entry in os.listdir(self._path):
            if len(entry) > 1 and entry[0] == '.' and \
               os.path.isdir(os.path.join(self._path, entry)):
                result.append(entry[1:])
        return result

    def get_folder(self, folder):
        """Return a Maildir instance for the named folder."""
        return Maildir(os.path.join(self._path, '.' + folder),
                       factory=self._factory,
                       create=False)

    def add_folder(self, folder):
        """Create a folder and return a Maildir instance representing it."""
        path = os.path.join(self._path, '.' + folder)
        result = Maildir(path, factory=self._factory)
        maildirfolder_path = os.path.join(path, 'maildirfolder')
        if not os.path.exists(maildirfolder_path):
            os.close(os.open(maildirfolder_path, os.O_CREAT | os.O_WRONLY,
                0o666))
        return result

    def remove_folder(self, folder):
        """Delete the named folder, which must be empty."""
        path = os.path.join(self._path, '.' + folder)
        for entry in os.listdir(os.path.join(path, 'new')) + \
                     os.listdir(os.path.join(path, 'cur')):
            if len(entry) < 1 or entry[0] != '.':
                raise NotEmptyError('Folder contains message(s): %s' % folder)
        for entry in os.listdir(path):
            if entry != 'new' and entry != 'cur' and entry != 'tmp' and \
               os.path.isdir(os.path.join(path, entry)):
                raise NotEmptyError("Folder contains subdirectory '%s': %s" %
                                    (folder, entry))
        for root, dirs, files in os.walk(path, topdown=False):
            for entry in files:
                os.remove(os.path.join(root, entry))
            for entry in dirs:
                os.rmdir(os.path.join(root, entry))
        os.rmdir(path)

    def clean(self):
        """Delete old files in "tmp"."""
        now = time.time()
        for entry in os.listdir(os.path.join(self._path, 'tmp')):
            path = os.path.join(self._path, 'tmp', entry)
            if now - os.path.getatime(path) > 129600:   # 60 * 60 * 36
                os.remove(path)

    _count = 1  # This is used to generate unique file names.

    def _create_tmp(self):
        """Create a file in the tmp subdirectory and open and return it."""
        now = time.time()
        hostname = socket.gethostname()
        if '/' in hostname:
            hostname = hostname.replace('/', r'\057')
        if ':' in hostname:
            hostname = hostname.replace(':', r'\072')
        uniq = "%s.M%sP%sQ%s.%s" % (int(now), int(now % 1 * 1e6), os.getpid(),
                                    Maildir._count, hostname)
        path = os.path.join(self._path, 'tmp', uniq)
        try:
            os.stat(path)
        except FileNotFoundError:
            Maildir._count += 1
            try:
                return _create_carefully(path)
            except FileExistsError:
                pass

        # Fall through to here if stat succeeded or open raised EEXIST.
        raise ExternalClashError('Name clash prevented file creation: %s' %
                                 path)

    def _refresh(self):
        """Update table of contents mapping."""
        # If it has been less than two seconds since the last _refresh() call,
        # we have to unconditionally re-read the mailbox just in case it has
        # been modified, because os.path.mtime() has a 2 sec resolution in the
        # most common worst case (FAT) and a 1 sec resolution typically.  This
        # results in a few unnecessary re-reads when _refresh() is called
        # multiple times in that interval, but once the clock ticks over, we
        # will only re-read as needed.  Because the filesystem might be being
        # served by an independent system with its own clock, we record and
        # compare with the mtimes from the filesystem.  Because the other
        # system's clock might be skewing relative to our clock, we add an
        # extra delta to our wait.  The default is one tenth second, but is an
        # instance variable and so can be adjusted if dealing with a
        # particularly skewed or irregular system.
        if time.time() - self._last_read > 2 + self._skewfactor:
            refresh = False
            for subdir in self._toc_mtimes:
                mtime = os.path.getmtime(self._paths[subdir])
                if mtime > self._toc_mtimes[subdir]:
                    refresh = True
                self._toc_mtimes[subdir] = mtime
            if not refresh:
                return
        # Refresh toc
        self._toc = {}
        for subdir in self._toc_mtimes:
            path = self._paths[subdir]
            for entry in os.listdir(path):
                p = os.path.join(path, entry)
                if os.path.isdir(p):
                    continue
                uniq = entry.split(self.colon)[0]
                self._toc[uniq] = os.path.join(subdir, entry)
        self._last_read = time.time()

    def _lookup(self, key):
        """Use TOC to return subpath for given key, or raise a KeyError."""
        try:
            if os.path.exists(os.path.join(self._path, self._toc[key])):
                return self._toc[key]
        except KeyError:
            pass
        self._refresh()
        try:
            return self._toc[key]
        except KeyError:
            raise KeyError('No message with key: %s' % key) from None

    # This method is for backward compatibility only.
    def next(self):
        """Return the next message in a one-time iteration."""
        if not hasattr(self, '_onetime_keys'):
            self._onetime_keys = self.iterkeys()
        while True:
            try:
                return self[next(self._onetime_keys)]
            except StopIteration:
                return None
            except KeyError:
                continue


class _singlefileMailbox(Mailbox):
    """A single-file mailbox."""

    def __init__(self, path, factory=None, create=True):
        """Initialize a single-file mailbox."""
        Mailbox.__init__(self, path, factory, create)
        try:
            f = open(self._path, 'rb+')
        except OSError as e:
            if e.errno == errno.ENOENT:
                if create:
                    f = open(self._path, 'wb+')
                else:
                    raise NoSuchMailboxError(self._path)
            elif e.errno in (errno.EACCES, errno.EROFS):
                f = open(self._path, 'rb')
            else:
                raise
        self._file = f
        self._toc = None
        self._next_key = 0
        self._pending = False       # No changes require rewriting the file.
        self._pending_sync = False  # No need to sync the file
        self._locked = False
        self._file_length = None    # Used to record mailbox size

    def add(self, message):
        """Add message and return assigned key."""
        self._lookup()
        self._toc[self._next_key] = self._append_message(message)
        self._next_key += 1
        # _append_message appends the message to the mailbox file. We
        # don't need a full rewrite + rename, sync is enough.
        self._pending_sync = True
        return self._next_key - 1

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        self._lookup(key)
        del self._toc[key]
        self._pending = True

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        self._lookup(key)
        self._toc[key] = self._append_message(message)
        self._pending = True

    def iterkeys(self):
        """Return an iterator over keys."""
        self._lookup()
        yield from self._toc.keys()

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        self._lookup()
        return key in self._toc

    def __len__(self):
        """Return a count of messages in the mailbox."""
        self._lookup()
        return len(self._toc)

    def lock(self):
        """Lock the mailbox."""
        if not self._locked:
            _lock_file(self._file)
            self._locked = True

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        if self._locked:
            _unlock_file(self._file)
            self._locked = False

    def flush(self):
        """Write any pending changes to disk."""
        if not self._pending:
            if self._pending_sync:
                # Messages have only been added, so syncing the file
                # is enough.
                _sync_flush(self._file)
                self._pending_sync = False
            return

        # In order to be writing anything out at all, self._toc must
        # already have been generated (and presumably has been modified
        # by adding or deleting an item).
        assert self._toc is not None

        # Check length of self._file; if it's changed, some other process
        # has modified the mailbox since we scanned it.
        self._file.seek(0, 2)
        cur_len = self._file.tell()
        if cur_len != self._file_length:
            raise ExternalClashError('Size of mailbox file changed '
                                     '(expected %i, found %i)' %
                                     (self._file_length, cur_len))

        new_file = _create_temporary(self._path)
        try:
            new_toc = {}
            self._pre_mailbox_hook(new_file)
            for key in sorted(self._toc.keys()):
                start, stop = self._toc[key]
                self._file.seek(start)
                self._pre_message_hook(new_file)
                new_start = new_file.tell()
                while True:
                    buffer = self._file.read(min(4096,
                                                 stop - self._file.tell()))
                    if not buffer:
                        break
                    new_file.write(buffer)
                new_toc[key] = (new_start, new_file.tell())
                self._post_message_hook(new_file)
            self._file_length = new_file.tell()
        except:
            new_file.close()
            os.remove(new_file.name)
            raise
        _sync_close(new_file)
        # self._file is about to get replaced, so no need to sync.
        self._file.close()
        # Make sure the new file's mode is the same as the old file's
        mode = os.stat(self._path).st_mode
        os.chmod(new_file.name, mode)
        try:
            os.rename(new_file.name, self._path)
        except FileExistsError:
            os.remove(self._path)
            os.rename(new_file.name, self._path)
        self._file = open(self._path, 'rb+')
        self._toc = new_toc
        self._pending = False
        self._pending_sync = False
        if self._locked:
            _lock_file(self._file, dotlock=False)

    def _pre_mailbox_hook(self, f):
        """Called before writing the mailbox to file f."""
        return

    def _pre_message_hook(self, f):
        """Called before writing each message to file f."""
        return

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        return

    def close(self):
        """Flush and close the mailbox."""
        try:
            self.flush()
        finally:
            try:
                if self._locked:
                    self.unlock()
            finally:
                self._file.close()  # Sync has been done by self.flush() above.

    def _lookup(self, key=None):
        """Return (start, stop) or raise KeyError."""
        if self._toc is None:
            self._generate_toc()
        if key is not None:
            try:
                return self._toc[key]
            except KeyError:
                raise KeyError('No message with key: %s' % key) from None

    def _append_message(self, message):
        """Append message to mailbox and return (start, stop) offsets."""
        self._file.seek(0, 2)
        before = self._file.tell()
        if len(self._toc) == 0 and not self._pending:
            # This is the first message, and the _pre_mailbox_hook
            # hasn't yet been called. If self._pending is True,
            # messages have been removed, so _pre_mailbox_hook must
            # have been called already.
            self._pre_mailbox_hook(self._file)
        try:
            self._pre_message_hook(self._file)
            offsets = self._install_message(message)
            self._post_message_hook(self._file)
        except BaseException:
            self._file.truncate(before)
            raise
        self._file.flush()
        self._file_length = self._file.tell()  # Record current length of mailbox
        return offsets



class _mboxMMDF(_singlefileMailbox):
    """An mbox or MMDF mailbox."""

    _mangle_from_ = True

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        from_line = self._file.readline().replace(linesep, b'')
        string = self._file.read(stop - self._file.tell())
        msg = self._message_factory(string.replace(linesep, b'\n'))
        msg.set_from(from_line[5:].decode('ascii'))
        return msg

    def get_string(self, key, from_=False):
        """Return a string representation or raise a KeyError."""
        return email.message_from_bytes(
            self.get_bytes(key, from_)).as_string(unixfrom=from_)

    def get_bytes(self, key, from_=False):
        """Return a string representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        if not from_:
            self._file.readline()
        string = self._file.read(stop - self._file.tell())
        return string.replace(linesep, b'\n')

    def get_file(self, key, from_=False):
        """Return a file-like representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        if not from_:
            self._file.readline()
        return _PartialFile(self._file, self._file.tell(), stop)

    def _install_message(self, message):
        """Format a message and blindly write to self._file."""
        from_line = None
        if isinstance(message, str):
            message = self._string_to_bytes(message)
        if isinstance(message, bytes) and message.startswith(b'From '):
            newline = message.find(b'\n')
            if newline != -1:
                from_line = message[:newline]
                message = message[newline + 1:]
            else:
                from_line = message
                message = b''
        elif isinstance(message, _mboxMMDFMessage):
            author = message.get_from().encode('ascii')
            from_line = b'From ' + author
        elif isinstance(message, email.message.Message):
            from_line = message.get_unixfrom()  # May be None.
            if from_line is not None:
                from_line = from_line.encode('ascii')
        if from_line is None:
            from_line = b'From MAILER-DAEMON ' + time.asctime(time.gmtime()).encode()
        start = self._file.tell()
        self._file.write(from_line + linesep)
        self._dump_message(message, self._file, self._mangle_from_)
        stop = self._file.tell()
        return (start, stop)


class mbox(_mboxMMDF):
    """A classic mbox mailbox."""

    _mangle_from_ = True

    # All messages must end in a newline character, and
    # _post_message_hooks outputs an empty line between messages.
    _append_newline = True

    def __init__(self, path, factory=None, create=True):
        """Initialize an mbox mailbox."""
        self._message_factory = mboxMessage
        _mboxMMDF.__init__(self, path, factory, create)

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        f.write(linesep)

    def _generate_toc(self):
        """Generate key-to-(start, stop) table of contents."""
        starts, stops = [], []
        last_was_empty = False
        self._file.seek(0)
        while True:
            line_pos = self._file.tell()
            line = self._file.readline()
            if line.startswith(b'From '):
                if len(stops) < len(starts):
                    if last_was_empty:
                        stops.append(line_pos - len(linesep))
                    else:
                        # The last line before the "From " line wasn't
                        # blank, but we consider it a start of a
                        # message anyway.
                        stops.append(line_pos)
                starts.append(line_pos)
                last_was_empty = False
            elif not line:
                if last_was_empty:
                    stops.append(line_pos - len(linesep))
                else:
                    stops.append(line_pos)
                break
            elif line == linesep:
                last_was_empty = True
            else:
                last_was_empty = False
        self._toc = dict(enumerate(zip(starts, stops)))
        self._next_key = len(self._toc)
        self._file_length = self._file.tell()


class MMDF(_mboxMMDF):
    """An MMDF mailbox."""

    def __init__(self, path, factory=None, create=True):
        """Initialize an MMDF mailbox."""
        self._message_factory = MMDFMessage
        _mboxMMDF.__init__(self, path, factory, create)

    def _pre_message_hook(self, f):
        """Called before writing each message to file f."""
        f.write(b'\001\001\001\001' + linesep)

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        f.write(linesep + b'\001\001\001\001' + linesep)

    def _generate_toc(self):
        """Generate key-to-(start, stop) table of contents."""
        starts, stops = [], []
        self._file.seek(0)
        next_pos = 0
        while True:
            line_pos = next_pos
            line = self._file.readline()
            next_pos = self._file.tell()
            if line.startswith(b'\001\001\001\001' + linesep):
                starts.append(next_pos)
                while True:
                    line_pos = next_pos
                    line = self._file.readline()
                    next_pos = self._file.tell()
                    if line == b'\001\001\001\001' + linesep:
                        stops.append(line_pos - len(linesep))
                        break
                    elif not line:
                        stops.append(line_pos)
                        break
            elif not line:
                break
        self._toc = dict(enumerate(zip(starts, stops)))
        self._next_key = len(self._toc)
        self._file.seek(0, 2)
        self._file_length = self._file.tell()


class MH(Mailbox):
    """An MH mailbox."""

    def __init__(self, path, factory=None, create=True):
        """Initialize an MH instance."""
        Mailbox.__init__(self, path, factory, create)
        if not os.path.exists(self._path):
            if create:
                os.mkdir(self._path, 0o700)
                os.close(os.open(os.path.join(self._path, '.mh_sequences'),
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            else:
                raise NoSuchMailboxError(self._path)
        self._locked = False

    def add(self, message):
        """Add message and return assigned key."""
        keys = self.keys()
        if len(keys) == 0:
            new_key = 1
        else:
            new_key = max(keys) + 1
        new_path = os.path.join(self._path, str(new_key))
        f = _create_carefully(new_path)
        closed = False
        try:
            if self._locked:
                _lock_file(f)
            try:
                try:
                    self._dump_message(message, f)
                except BaseException:
                    # Unlock and close so it can be deleted on Windows
                    if self._locked:
                        _unlock_file(f)
                    _sync_close(f)
                    closed = True
                    os.remove(new_path)
                    raise
