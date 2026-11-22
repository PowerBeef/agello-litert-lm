// Copyright 2025 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Recursive-descent parser for Gemma FC (function-calling) expressions.
//!
//! Grammar:
//! ```text
//! start         := ws (function_call (ws function_call)*)? ws EOF
//! function_call := "call" ws ":" ws identifier ws object?
//! object        := "{" ws (pair (ws "," ws pair)*)? ws "}"
//! pair          := key ws ":" ws value
//! key           := identifier | delimited_string | json_string
//! array         := "[" ws (value (ws "," ws value)*)? ws "]"
//! value         := delimited_string
//!                | json_string
//!                | number
//!                | boolean
//!                | null_literal
//!                | object
//!                | array
//! ```
//!
//! Strings support both paired FC control delimiters (`<|"|>`, `<escape>`,
//! and the raw ctrl46 delimiter — scanned verbatim, preserving literal quotes
//! and backslashes) and standard RFC 8259 JSON double-quoted strings (`"..."`),
//! which Gemma 4 occasionally emits in place of `<|"|>` for complex string
//! arguments.

use serde_json::{Map, Number, Value};
use std::collections::HashSet;

const MAX_NESTING_DEPTH: usize = 128;

const DELIMITERS: [&str; 3] = ["<|\"|>", "<escape>", concat!("<", "ctrl46", ">")];

fn is_id_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_id_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')
}

struct Parser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, bytes: input.as_bytes(), pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(format!("Expected '{}' at byte {}", expected as char, self.pos))
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        if !self.rest().starts_with(keyword) {
            return false;
        }
        let end = self.pos + keyword.len();
        if let Some(&next) = self.bytes.get(end) {
            if is_id_continue(next) {
                return false;
            }
        }
        self.pos = end;
        true
    }

    fn peek_delimiter(&self) -> Option<&'static str> {
        let rest = self.rest();
        DELIMITERS.into_iter().find(|&d| rest.starts_with(d))
    }

    fn parse_identifier(&mut self) -> Option<&'a str> {
        let start = self.pos;
        if !matches!(self.peek(), Some(b) if is_id_start(b)) {
            return None;
        }
        self.pos += 1;
        while matches!(self.peek(), Some(b) if is_id_continue(b)) {
            self.pos += 1;
        }
        Some(&self.input[start..self.pos])
    }

    fn parse_delimited_string(&mut self, delim: &str) -> Result<String, String> {
        let start = self.pos;
        debug_assert!(self.rest().starts_with(delim));
        self.pos += delim.len();
        let body_start = self.pos;
        let Some(rel_end) = self.rest().find(delim) else {
            return Err(format!("Unterminated {} string starting at byte {}", delim, start));
        };
        let content = &self.input[body_start..body_start + rel_end];
        self.pos = body_start + rel_end + delim.len();
        Ok(content.to_string())
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        if self.bytes.len() - self.pos < 4 {
            return Err(format!("Incomplete unicode escape at byte {}", self.pos));
        }
        let mut value: u16 = 0;
        for &b in &self.bytes[self.pos..self.pos + 4] {
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a' + 10) as u16,
                b'A'..=b'F' => (b - b'A' + 10) as u16,
                _ => {
                    return Err(format!("Invalid hex digit in unicode escape at byte {}", self.pos));
                }
            };
            value = (value << 4) | digit;
        }
        self.pos += 4;
        Ok(value)
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        let first = self.parse_hex4()?;
        if (0xD800..=0xDBFF).contains(&first) {
            if !self.rest().starts_with("\\u") {
                return Err(format!(
                    "Unpaired high surrogate \\u{:04X} at byte {}",
                    first, self.pos
                ));
            }
            self.pos += 2;
            let second = self.parse_hex4()?;
            if !(0xDC00..=0xDFFF).contains(&second) {
                return Err(format!(
                    "Expected low surrogate after \\u{:04X}, got \\u{:04X} at byte {}",
                    first, second, self.pos
                ));
            }
            let code_point =
                0x10000 + (((first as u32 - 0xD800) << 10) | (second as u32 - 0xDC00));
            char::from_u32(code_point)
                .ok_or_else(|| format!("Invalid surrogate pair at byte {}", self.pos))
        } else if (0xDC00..=0xDFFF).contains(&first) {
            Err(format!("Unpaired low surrogate \\u{:04X} at byte {}", first, self.pos))
        } else {
            char::from_u32(first as u32).ok_or_else(|| {
                format!("Invalid unicode scalar \\u{:04X} at byte {}", first, self.pos)
            })
        }
    }

    fn parse_json_string(&mut self) -> Result<String, String> {
        let start_quote = self.pos;
        self.expect_byte(b'"')?;
        let mut out = String::new();
        let mut chunk_start = self.pos;

        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b'"' => {
                    out.push_str(&self.input[chunk_start..self.pos]);
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    out.push_str(&self.input[chunk_start..self.pos]);
                    self.pos += 1;
                    let escaped = self.rest().chars().next().ok_or_else(|| {
                        format!(
                            "Unterminated escape sequence in string starting at byte {}",
                            start_quote
                        )
                    })?;
                    self.pos += escaped.len_utf8();
                    match escaped {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{0008}'),
                        'f' => out.push('\u{000C}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let ch = self.parse_unicode_escape()?;
                            out.push(ch);
                        }
                        other => {
                            // Preserve non-JSON backslash escapes (e.g., regex or path
                            // sequences like `\d`, `\s`) verbatim rather than failing.
                            out.push('\\');
                            out.push(other);
                        }
                    }
                    chunk_start = self.pos;
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
        Err(format!("Unterminated double-quoted string starting at byte {}", start_quote))
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        let mut has_frac = false;
        let mut has_exp = false;

        if self.peek() == Some(b'-') {
            self.pos += 1;
        }

        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            Some(b'.') => {}
            _ => {
                return Err(format!("Invalid number at byte {}", start));
            }
        }

        if self.peek() == Some(b'.') {
            has_frac = true;
            self.pos += 1;
            let frac_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == frac_start {
                return Err(format!("Expected digit after '.' in number at byte {}", start));
            }
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            has_exp = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(format!("Expected digit in exponent at byte {}", start));
            }
        }

        // Reject malformed trailing characters such as `1.2.3`, `01`, or `12abc`.
        if let Some(next) = self.peek() {
            if next == b'.' || next.is_ascii_alphanumeric() || next == b'_' {
                return Err(format!("Malformed number at byte {}", start));
            }
        }

        let text = &self.input[start..self.pos];
        // Branch on the literal text so integers keep exact i64 representation
        // (and precision above 2^53) instead of widening to f64.
        if !has_frac && !has_exp {
            if let Ok(int_val) = text.parse::<i64>() {
                return Ok(Value::Number(Number::from(int_val)));
            }
        }
        let float_val =
            text.parse::<f64>().map_err(|_| format!("Failed to parse number: {}", text))?;
        Number::from_f64(float_val)
            .map(Value::Number)
            .ok_or_else(|| format!("Number out of range: {}", text))
    }

    fn parse_object_key(&mut self) -> Result<String, String> {
        if let Some(delim) = self.peek_delimiter() {
            return self.parse_delimited_string(delim);
        }
        if self.peek() == Some(b'"') {
            return self.parse_json_string();
        }
        self.parse_identifier()
            .map(str::to_string)
            .ok_or_else(|| format!("Expected object key at byte {}", self.pos))
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value, String> {
        if depth >= MAX_NESTING_DEPTH {
            return Err(format!(
                "Maximum nesting depth ({}) exceeded at byte {}",
                MAX_NESTING_DEPTH, self.pos
            ));
        }
        self.expect_byte(b'{')?;
        self.skip_ws();
        let mut map = Map::new();
        let mut seen_keys = HashSet::new();

        if self.consume_byte(b'}') {
            return Ok(Value::Object(map));
        }

        loop {
            let key_pos = self.pos;
            let key = self.parse_object_key()?;
            if key.is_empty() {
                return Err(format!("Object key is empty at byte {}", key_pos));
            }
            self.skip_ws();
            self.expect_byte(b':')?;
            self.skip_ws();
            let value = self
                .parse_value(depth + 1)
                .map_err(|e| format!("Error parsing value for key '{}': {}", key, e))?;

            if seen_keys.insert(key.clone()) {
                map.insert(key, value);
            }

            self.skip_ws();
            if self.consume_byte(b',') {
                self.skip_ws();
                continue;
            }
            if self.consume_byte(b'}') {
                return Ok(Value::Object(map));
            }
            return Err(format!("Expected ',' or '}}' in object at byte {}", self.pos));
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value, String> {
        if depth >= MAX_NESTING_DEPTH {
            return Err(format!(
                "Maximum nesting depth ({}) exceeded at byte {}",
                MAX_NESTING_DEPTH, self.pos
            ));
        }
        self.expect_byte(b'[')?;
        self.skip_ws();
        let mut list = Vec::new();

        if self.consume_byte(b']') {
            return Ok(Value::Array(list));
        }

        loop {
            list.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            if self.consume_byte(b',') {
                self.skip_ws();
                continue;
            }
            if self.consume_byte(b']') {
                return Ok(Value::Array(list));
            }
            return Err(format!("Expected ',' or ']' in array at byte {}", self.pos));
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value, String> {
        if let Some(delim) = self.peek_delimiter() {
            return self.parse_delimited_string(delim).map(Value::String);
        }
        match self.peek() {
            Some(b'"') => self.parse_json_string().map(Value::String),
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'-' | b'.' | b'0'..=b'9') => self.parse_number(),
            Some(b't') if self.consume_keyword("true") => Ok(Value::Bool(true)),
            Some(b'f') if self.consume_keyword("false") => Ok(Value::Bool(false)),
            Some(b'n') if self.consume_keyword("null") => Ok(Value::Null),
            Some(_) => Err(format!("Unexpected token at byte {}", self.pos)),
            None => Err(format!("Unexpected end of input while parsing value at byte {}", self.pos)),
        }
    }

    fn parse_function_call(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if !self.consume_keyword("call") {
            return Err(format!("Expected 'call:' at byte {}", start));
        }
        self.skip_ws();
        if !self.consume_byte(b':') {
            return Err(format!("Expected ':' after 'call' at byte {}", self.pos));
        }
        self.skip_ws();
        let name = self
            .parse_identifier()
            .ok_or_else(|| format!("Expected tool name at byte {}", self.pos))?;
        self.skip_ws();
        let arguments =
            if self.peek() == Some(b'{') { self.parse_object(0)? } else { Value::Object(Map::new()) };

        let mut call = Map::new();
        call.insert("name".to_string(), Value::String(name.to_string()));
        call.insert("arguments".to_string(), arguments);
        Ok(Value::Object(call))
    }
}

pub fn parse_fc_expression(text: &str) -> Result<Vec<Value>, String> {
    let mut parser = Parser::new(text);
    parser.skip_ws();
    if parser.is_eof() {
        return Ok(Vec::new());
    }
    let mut tool_calls = Vec::new();
    while !parser.is_eof() {
        tool_calls.push(parser.parse_function_call()?);
        parser.skip_ws();
    }
    Ok(tool_calls)
}

#[cfg(test)]
mod fc_parser_test;
