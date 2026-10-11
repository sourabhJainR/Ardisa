use crate::source::{Diagnostic, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Module,
    Fn,
    If,
    Else,
    Let,
    Set,
    Return,
    Scope,
    Spawn,
    Join,
    Cancel,
    While,
    True,
    False,
    Ident,
    Int,
    Float,
    String,
    Arrow,
    Equal,
    EqualEqual,
    NotEqual,
    LessEqual,
    GreaterEqual,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Comma,
    Colon,
    LBracket,
    RBracket,
    LParen,
    RParen,
    LAngle,
    RAngle,
    Newline,
    Indent(u16),
    Dedent,
    Eof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub lexeme: String,
}

pub fn lex(input: &str) -> Result<Vec<Token>, Vec<Diagnostic>> {
    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    let mut indent_stack = vec![0u16];
    let mut offset = 0usize;

    for line in input.split_inclusive('\n') {
        let line_start = offset;
        offset += line.len();
        let content = line.strip_suffix('\n').unwrap_or(line);
        let newline_span = Span::new(line_start + content.len(), line_start + line.len());

        if content.trim().is_empty() {
            continue;
        }

        let bytes = content.as_bytes();
        let mut index = 0usize;
        let mut spaces = 0u16;
        while index < bytes.len() && bytes[index] == b' ' {
            spaces = spaces.saturating_add(1);
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b'\t' {
            errors.push(Diagnostic::error(
                "AIF100",
                "tabs are not allowed for indentation; use spaces",
                Some(Span::new(line_start, line_start + index + 1)),
            ));
            continue;
        }

        let current = *indent_stack.last().unwrap_or(&0);
        if spaces > current {
            indent_stack.push(spaces);
            tokens.push(Token {
                kind: TokenKind::Indent(spaces),
                span: Span::new(line_start, line_start + spaces as usize),
                lexeme: content[..spaces as usize].to_owned(),
            });
        } else if spaces < current {
            while spaces < *indent_stack.last().unwrap_or(&0) {
                indent_stack.pop();
                tokens.push(Token {
                    kind: TokenKind::Dedent,
                    span: Span::new(line_start, line_start + spaces as usize),
                    lexeme: String::new(),
                });
            }
            if spaces != *indent_stack.last().unwrap_or(&0) {
                errors.push(Diagnostic::error(
                    "AIF101",
                    "indentation does not match an outer block",
                    Some(Span::new(line_start, line_start + spaces as usize)),
                ));
            }
        }

        let mut cursor = index;
        while cursor < bytes.len() {
            let c = bytes[cursor] as char;
            if c == ' ' {
                cursor += 1;
                continue;
            }
            if c == '#' {
                break;
            }

            let start = cursor;
            if c.is_ascii_alphabetic() || c == '_' {
                cursor += 1;
                while cursor < bytes.len() {
                    let next = bytes[cursor] as char;
                    if next.is_ascii_alphanumeric() || next == '_' {
                        cursor += 1;
                    } else {
                        break;
                    }
                }
                let word = &content[start..cursor];
                let kind = match word {
                    "module" => TokenKind::Module,
                    "fn" => TokenKind::Fn,
                    "if" => TokenKind::If,
                    "else" => TokenKind::Else,
                    "let" => TokenKind::Let,
                    "set" => TokenKind::Set,
                    "return" => TokenKind::Return,
                    "scope" => TokenKind::Scope,
                    "spawn" => TokenKind::Spawn,
                    "join" => TokenKind::Join,
                    "cancel" => TokenKind::Cancel,
                    "while" => TokenKind::While,
                    "true" => TokenKind::True,
                    "false" => TokenKind::False,
                    _ => TokenKind::Ident,
                };
                tokens.push(Token {
                    kind,
                    span: Span::new(line_start + start, line_start + cursor),
                    lexeme: word.into(),
                });
                continue;
            }

            if c.is_ascii_digit() {
                cursor += 1;
                while cursor < bytes.len() && (bytes[cursor] as char).is_ascii_digit() {
                    cursor += 1;
                }
                let mut kind = TokenKind::Int;
                if cursor + 1 < bytes.len()
                    && bytes[cursor] == b'.'
                    && (bytes[cursor + 1] as char).is_ascii_digit()
                {
                    kind = TokenKind::Float;
                    cursor += 1;
                    while cursor < bytes.len() && (bytes[cursor] as char).is_ascii_digit() {
                        cursor += 1;
                    }
                }
                let word = &content[start..cursor];
                tokens.push(Token {
                    kind,
                    span: Span::new(line_start + start, line_start + cursor),
                    lexeme: word.into(),
                });
                continue;
            }

            if c == '"' {
                cursor += 1;
                let mut escaped = false;
                let mut closed = false;
                while cursor < bytes.len() {
                    let next = bytes[cursor] as char;
                    cursor += 1;
                    if escaped {
                        escaped = false;
                    } else if next == '\\' {
                        escaped = true;
                    } else if next == '"' {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    errors.push(Diagnostic::error(
                        "AIF102",
                        "unterminated string literal",
                        Some(Span::new(line_start + start, line_start + cursor)),
                    ));
                } else {
                    tokens.push(Token {
                        kind: TokenKind::String,
                        span: Span::new(line_start + start, line_start + cursor),
                        lexeme: content[start + 1..cursor - 1].into(),
                    });
                }
                continue;
            }

            let (kind, width) = if content[start..].starts_with("->") {
                (TokenKind::Arrow, 2)
            } else if content[start..].starts_with("==") {
                (TokenKind::EqualEqual, 2)
            } else if content[start..].starts_with("!=") {
                (TokenKind::NotEqual, 2)
            } else if content[start..].starts_with("<=") {
                (TokenKind::LessEqual, 2)
            } else if content[start..].starts_with(">=") {
                (TokenKind::GreaterEqual, 2)
            } else {
                match c {
                    '=' => (TokenKind::Equal, 1),
                    '+' => (TokenKind::Plus, 1),
                    '-' => (TokenKind::Minus, 1),
                    '*' => (TokenKind::Star, 1),
                    '/' => (TokenKind::Slash, 1),
                    '%' => (TokenKind::Percent, 1),
                    ',' => (TokenKind::Comma, 1),
                    ':' => (TokenKind::Colon, 1),
                    '[' => (TokenKind::LBracket, 1),
                    ']' => (TokenKind::RBracket, 1),
                    '(' => (TokenKind::LParen, 1),
                    ')' => (TokenKind::RParen, 1),
                    '<' => (TokenKind::LAngle, 1),
                    '>' => (TokenKind::RAngle, 1),
                    _ => {
                        errors.push(Diagnostic::error(
                            "AIF103",
                            format!("unexpected character '{c}'"),
                            Some(Span::new(line_start + start, line_start + start + 1)),
                        ));
                        cursor += 1;
                        continue;
                    }
                }
            };
            cursor += width;
            tokens.push(Token {
                kind,
                span: Span::new(line_start + start, line_start + cursor),
                lexeme: content[start..cursor].into(),
            });
        }

        tokens.push(Token {
            kind: TokenKind::Newline,
            span: newline_span,
            lexeme: String::new(),
        });
    }

    while indent_stack.len() > 1 {
        indent_stack.pop();
        tokens.push(Token {
            kind: TokenKind::Dedent,
            span: Span::new(input.len(), input.len()),
            lexeme: String::new(),
        });
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(input.len(), input.len()),
        lexeme: String::new(),
    });

    if errors.is_empty() {
        Ok(tokens)
    } else {
        Err(errors)
    }
}
