/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tinygpt::tokenizer {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct Value;
using ValueList = std::vector<Value>;

struct Value {
  enum Type {
    NONE,
    BOOL,
    INT,
    STRING,
    MESSAGE,
    MESSAGE_LIST,
    STRING_LIST
  };

  std::variant<std::monostate, bool, int64_t, std::string, ChatMessage, std::vector<ChatMessage>,
               std::vector<std::string>>
      data;

  Value() : data(std::monostate{}) {}
  explicit Value(bool v) : data(v) {}
  explicit Value(int64_t v) : data(v) {}
  explicit Value(const std::string& v) : data(v) {}
  explicit Value(std::string&& v) : data(std::move(v)) {}
  explicit Value(const ChatMessage& v) : data(v) {}
  explicit Value(const std::vector<ChatMessage>& v) : data(v) {}
  explicit Value(const std::vector<std::string>& v) : data(v) {}
  explicit Value(std::vector<std::string>&& v) : data(std::move(v)) {}

  Type type() const { return static_cast<Type>(data.index()); }
  bool isNone() const { return type() == NONE; }
  bool isBool() const { return type() == BOOL; }
  bool isInt() const { return type() == INT; }
  bool isString() const { return type() == STRING; }
  bool isMessage() const { return type() == MESSAGE; }
  bool isMessageList() const { return type() == MESSAGE_LIST; }
  bool isStringList() const { return type() == STRING_LIST; }

  bool asBool() const { return std::get<bool>(data); }
  int64_t asInt() const { return std::get<int64_t>(data); }
  const std::string& asString() const { return std::get<std::string>(data); }
  const ChatMessage& asMessage() const { return std::get<ChatMessage>(data); }
  const std::vector<ChatMessage>& asMessageList() const { return std::get<std::vector<ChatMessage>>(data); }
  const std::vector<std::string>& asStringList() const { return std::get<std::vector<std::string>>(data); }

  // truthiness: none=false, bool=value, int=!=0, string=!empty, message=true, list=!empty
  bool truthy() const;

  std::string toString() const;
};

enum class TokenType {
  TEXT,         // literal text outside {{ }} / {% %}
  VAR_BEGIN,    // {{
  VAR_END,      // }}
  BLOCK_BEGIN,  // {%
  BLOCK_END,    // %}
  STRING_LIT,   // 'string' or "string"
  INT_LIT,      // integer literal
  IDENTIFIER,   // variable name / keyword
  DOT,          // .
  LBRACKET,     // [
  RBRACKET,     // ]
  LPAREN,       // (
  RPAREN,       // )
  PIPE,         // |
  COMMA,        // ,
  EQ,           // ==
  NEQ,          // !=
  ASSIGN,       // =
  PLUS,         // +
  MINUS,        // -
  LT,           // <
  GT,           // >
  LTE,          // <=
  GTE,          // >=
  MODULO,       // %
  COLON,        // :
  TILDE,        // ~  (string concatenation in Jinja2)
  END_OF_INPUT
};

struct Token {
  TokenType type;
  std::string value;
};

struct Expr;
struct Stmt;

using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

struct Expr {
  enum Kind {
    STRING_LITERAL,
    BOOL_LITERAL,
    INT_LITERAL,
    NONE_LITERAL,
    IDENTIFIER,
    MEMBER_ACCESS,  // expr.member
    INDEX_ACCESS,   // expr['key'] or expr[index]
    BINARY_OP,      // expr op expr
    UNARY_OP,       // not expr
    FILTER,         // expr | filter
    FUNC_CALL,      // func(args)
    METHOD_CALL,    // expr.method(args)
    SLICE_ACCESS,   // expr[start:stop:step]
  } kind;

  std::string strValue;
  bool boolValue = false;
  int64_t intValue = 0;

  ExprPtr left;
  ExprPtr right;
  std::string op;  // operator or member name or filter name

  std::vector<ExprPtr> args;                            // function/method call arguments
  std::vector<std::pair<std::string, ExprPtr>> kwargs;  // keyword arguments
  std::vector<ExprPtr> sliceArgs;                       // slice: [start, stop, step] (nullptr = omitted)
};

struct Stmt {
  enum Kind {
    TEXT,
    PRINT,  // {{ expr }}
    IF,     // if / elif / else / endif
    FOR,    // for var in expr / endfor
    SET,    // set var = expr
  } kind;

  std::string textContent;

  // PRINT
  ExprPtr printExpr;

  // IF: conditions[i] -> bodies[i], last body may be else (condition=nullptr)
  struct IfBranch {
    ExprPtr condition;  // nullptr for else branch
    std::vector<StmtPtr> body;
  };
  std::vector<IfBranch> branches;

  // FOR
  std::string loopVar;
  ExprPtr iterExpr;
  std::vector<StmtPtr> forBody;

  // SET
  std::string setVar;
  ExprPtr setExpr;
};

class ChatTemplateEngine {
 public:
  static std::string render(const std::string& tmpl, const std::vector<ChatMessage>& messages, bool addGenerationPrompt,
                            const std::string& bosToken, const std::string& eosToken);

 private:
  // Lexer
  static std::vector<Token> tokenize(const std::string& tmpl);
  static std::vector<Token> tokenizeExpr(const std::string& content);

  // Parser
  static std::vector<StmtPtr> parse(const std::vector<Token>& tokens);

  // Evaluator
  struct EvalContext {
    std::vector<std::unordered_map<std::string, Value>> scopes;

    void pushScope();
    void popScope();
    void set(const std::string& name, Value value);
    Value get(const std::string& name) const;
  };

  static Value evalExpr(const Expr& expr, EvalContext& ctx);
  static void evalStmt(const Stmt& stmt, EvalContext& ctx, std::string& output);
  static void evalStmts(const std::vector<StmtPtr>& stmts, EvalContext& ctx, std::string& output);
};

std::string applyChatTemplate(const std::string& tmpl, const std::vector<ChatMessage>& messages,
                              bool addGenerationPrompt = true, const std::string& bosToken = "",
                              const std::string& eosToken = "");

}  // namespace tinygpt::tokenizer
