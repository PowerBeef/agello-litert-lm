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

use super::*;

/// Parses `call:test_tool{value:<literal>}` and returns the `value` argument.
fn parse_value_literal(literal: &str) -> Value {
    let text = format!("call:test_tool{{value:{}}}", literal);
    let tool_calls = parse_fc_expression(&text).expect("parse failed");
    assert_eq!(tool_calls.len(), 1);
    tool_calls[0]["arguments"]["value"].clone()
}

#[test]
fn parses_integers_as_integers() {
    // The grammar has a single NUMBER token for both integers and floats, so
    // the type is decided from the literal text. Parsing everything as f64
    // would turn 1000 into 1000.0, which a strict consumer rejects for a field
    // declared as an integer.
    for literal in ["1000", "-678", "0"] {
        let value = parse_value_literal(literal);
        assert!(value.is_i64(), "{} should be an integer, got {}", literal, value);
        assert_eq!(value.to_string(), literal);
    }
}

#[test]
fn parses_floats_as_floats() {
    for literal in ["1000.0", "3.14", "-4.5"] {
        let value = parse_value_literal(literal);
        assert!(value.is_f64(), "{} should be a float, got {}", literal, value);
    }
}

#[test]
fn keeps_exponent_notation_as_float() {
    // The writer chose exponent notation and there is no way to tell an
    // intended integer from an intended float, so stay conservative.
    let value = parse_value_literal("1e3");
    assert!(value.is_f64());
    assert_eq!(value.as_f64(), Some(1000.0));
}

#[test]
fn preserves_integers_beyond_f64_precision() {
    // Above 2^53 an f64 round trip silently changes the digits.
    let value = parse_value_literal("9007199254740993");
    assert_eq!(value.as_i64(), Some(9007199254740993));
    assert_eq!(value.to_string(), "9007199254740993");
}

#[test]
fn preserves_integers_nested_in_containers() {
    let tool_calls =
        parse_fc_expression("call:t{list:[10,20],obj:{inner:30}}").expect("parse failed");
    let args = &tool_calls[0]["arguments"];
    assert!(args["list"][0].is_i64());
    assert!(args["list"][1].is_i64());
    assert!(args["obj"]["inner"].is_i64());
}

#[test]
fn rejects_malformed_numbers() {
    // Falling through both integer and float parsing is still an error.
    assert!(parse_fc_expression("call:t{value:1.2.3}").is_err());
    assert!(parse_fc_expression("call:t{value:01}").is_err());
    assert!(parse_fc_expression("call:t{value:1.}").is_err());
    assert!(parse_fc_expression("call:t{value:1e}").is_err());
    assert!(parse_fc_expression("call:t{value:-}").is_err());
}

#[test]
fn parses_scientific_notation_with_fraction() {
    let value = parse_value_literal("1.5e-3");
    assert!(value.is_f64());
    assert!((value.as_f64().unwrap() - 0.0015).abs() < 1e-12);
}

#[test]
fn parses_json_double_quoted_strings() {
    let expr = r#"call:calculate{expression:"250 * (((1 + (0.04 / 12))^60 - 1) / (0.04 / 12))"}"#;
    let tool_calls = parse_fc_expression(expr).expect("parse failed");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["name"], "calculate");
    assert_eq!(
        tool_calls[0]["arguments"]["expression"],
        "250 * (((1 + (0.04 / 12))^60 - 1) / (0.04 / 12))"
    );
}

#[test]
fn parses_json_string_escapes_and_unicode() {
    let expr = r#"call:t{s:"line1\nline2\t\"quoted\" \\slash\/ \u00e9 \uD83D\uDE00",raw:"\d+"}"#;
    let tool_calls = parse_fc_expression(expr).expect("parse failed");
    let args = &tool_calls[0]["arguments"];
    assert_eq!(args["s"], "line1\nline2\t\"quoted\" \\slash/ é 😀");
    assert_eq!(args["raw"], r"\d+");
}

#[test]
fn preserves_embedded_quotes_in_delimited_strings() {
    let expr = r#"call:search{q:<|"|>say "hello" and 'world'<|"|>,b:<escape>a \b\c "d"<escape>}"#;
    let tool_calls = parse_fc_expression(expr).expect("parse failed");
    let args = &tool_calls[0]["arguments"];
    assert_eq!(args["q"], r#"say "hello" and 'world'"#);
    assert_eq!(args["b"], r#"a \b\c "d""#);
}

#[test]
fn parses_ctrl46_delimited_strings_and_enforces_paired_delimiters() {
    let d = concat!("<", "ctrl46", ">");
    let ok_expr = format!("call:t{{s:{d}hello world{d}}}");
    let tool_calls = parse_fc_expression(&ok_expr).expect("ctrl46 parse failed");
    assert_eq!(tool_calls[0]["arguments"]["s"], "hello world");

    let mismatched = format!("call:t{{s:<escape>hello{d}}}");
    assert!(parse_fc_expression(&mismatched).is_err());

    let nested_other = format!(r#"call:t{{s:<|"|>keep <escape>tag<escape> intact<|"|>}}"#);
    let tool_calls = parse_fc_expression(&nested_other).expect("nested other delim failed");
    assert_eq!(tool_calls[0]["arguments"]["s"], "keep <escape>tag<escape> intact");
}

#[test]
fn parses_mixed_delimiters_and_quoted_keys() {
    let expr = r#"call:t{"first":"one",<|"|>second<|"|>:<|"|>two<|"|>,arr:["a",<|"|>b<|"|>,{"k":"v"}]}"#;
    let tool_calls = parse_fc_expression(expr).expect("parse failed");
    let args = &tool_calls[0]["arguments"];
    assert_eq!(args["first"], "one");
    assert_eq!(args["second"], "two");
    assert_eq!(args["arr"][0], "a");
    assert_eq!(args["arr"][1], "b");
    assert_eq!(args["arr"][2]["k"], "v");
}

#[test]
fn parses_multiple_calls_in_one_block() {
    let expr = r#"call:a{x:"one"} call:b{y:<|"|>two<|"|>}"#;
    let tool_calls = parse_fc_expression(expr).expect("parse failed");
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0]["name"], "a");
    assert_eq!(tool_calls[0]["arguments"]["x"], "one");
    assert_eq!(tool_calls[1]["name"], "b");
    assert_eq!(tool_calls[1]["arguments"]["y"], "two");
}

#[test]
fn rejects_unquoted_unterminated_and_malformed_strings() {
    assert!(parse_fc_expression(r#"call:print{text:hello}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{text:"unterminated}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{text:<|"|>unterminated}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{text:"bad \u12"}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{text:"bad \uD800"}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{text:"bad \uDC00"}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{text:"bad \u😀"}"#).is_err());
    assert!(parse_fc_expression(r#"call:print{"":1}"#).is_err());
}

#[test]
fn rejects_excessive_nesting_depth() {
    let open = "[".repeat(200);
    let close = "]".repeat(200);
    let expr = format!("call:t{{a:{open}1{close}}}");
    assert!(parse_fc_expression(&expr).is_err());
}
