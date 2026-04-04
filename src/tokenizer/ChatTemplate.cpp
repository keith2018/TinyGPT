/*
 * TinyGPT
 * @author 	: keith@robot9.me
 *
 */

#include "ChatTemplate.h"

#include <algorithm>
#include <string>

#include "Utils/Logger.h"

namespace tinygpt::tokenizer {

bool Value::truthy() const {
  switch (type()) {
    case NONE:
      return false;
    case BOOL:
      return asBool();
    case INT:
      return asInt() != 0;
    case STRING:
      return !asString().empty();
    case MESSAGE:
      return true;
    case MESSAGE_LIST:
      return !asMessageList().empty();
    case STRING_LIST:
      return !asStringList().empty();
  }
  return false;
}

std::string Value::toString() const {
  switch (type()) {
    case NONE:
      return "";
    case BOOL:
      return asBool() ? "True" : "False";
    case INT:
      return std::to_string(asInt());
    case STRING:
      return asString();
    case MESSAGE:
      return "[ChatMessage]";
    case MESSAGE_LIST:
      return "[ChatMessageList]";
    case STRING_LIST:
      return "[StringList]";
  }
  return "";
}

void ChatTemplateEngine::EvalContext::pushScope() { scopes.emplace_back(); }

void ChatTemplateEngine::EvalContext::popScope() {
  if (!scopes.empty()) scopes.pop_back();
}

void ChatTemplateEngine::EvalContext::set(const std::string& name, Value value) {
  if (scopes.empty()) return;

  // handle "ns.member" style set: find "ns" in any scope and update "ns.member" there
  auto dotPos = name.find('.');
  if (dotPos != std::string::npos) {
    std::string nsName = name.substr(0, dotPos);
    // find the scope that contains the namespace variable
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
      if (it->find(nsName) != it->end()) {
        (*it)[name] = std::move(value);
        return;
      }
    }
  }

  scopes.back()[name] = std::move(value);
}

Value ChatTemplateEngine::EvalContext::get(const std::string& name) const {
  // search from innermost scope
  for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
    auto found = it->find(name);
    if (found != it->end()) {
      return found->second;
    }
  }
  return {};  // none
}

static std::string trimWhitespace(const std::string& s) {
  size_t start = 0, end = s.size();
  while (start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
  while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\n' || s[end - 1] == '\r')) end--;
  return s.substr(start, end - start);
}

static void rtrimWhitespace(std::string& s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
}

std::vector<Token> ChatTemplateEngine::tokenize(const std::string& tmpl) {
  std::vector<Token> tokens;
  size_t pos = 0;
  size_t len = tmpl.size();

  while (pos < len) {
    // check for {{ or {%
    if (pos + 1 < len && tmpl[pos] == '{') {
      bool isVar = (tmpl[pos + 1] == '{');
      bool isBlock = (tmpl[pos + 1] == '%');

      if (isVar || isBlock) {
        // check for whitespace control: {{- or {%-
        bool trimLeft = (pos + 2 < len && tmpl[pos + 2] == '-');
        size_t tagStart = pos + 2 + (trimLeft ? 1 : 0);

        // find closing tag
        const char* closeTag = isVar ? "}}" : "%}";
        size_t closePos = tmpl.find(closeTag, tagStart);
        if (closePos == std::string::npos) {
          LOGE("Unclosed template tag at pos %zu", pos);
          tokens.push_back({TokenType::TEXT, tmpl.substr(pos)});
          break;
        }

        // check for whitespace control on close: -}} or -%}
        bool trimRight = (closePos > 0 && tmpl[closePos - 1] == '-');
        size_t contentEnd = trimRight ? closePos - 1 : closePos;

        // trim left whitespace from previous text token
        if (trimLeft && !tokens.empty() && tokens.back().type == TokenType::TEXT) {
          rtrimWhitespace(tokens.back().value);
          if (tokens.back().value.empty()) {
            tokens.pop_back();
          }
        }

        std::string content = tmpl.substr(tagStart, contentEnd - tagStart);

        // emit tag tokens
        if (isVar) {
          tokens.push_back({TokenType::VAR_BEGIN, "{{"});
          auto exprTokens = tokenizeExpr(content);
          tokens.insert(tokens.end(), exprTokens.begin(), exprTokens.end());
          tokens.push_back({TokenType::VAR_END, "}}"});
        } else {
          tokens.push_back({TokenType::BLOCK_BEGIN, "{%"});
          auto exprTokens = tokenizeExpr(content);
          tokens.insert(tokens.end(), exprTokens.begin(), exprTokens.end());
          tokens.push_back({TokenType::BLOCK_END, "%}"});
        }

        pos = closePos + 2;

        // trim right whitespace
        if (trimRight) {
          while (pos < len && (tmpl[pos] == ' ' || tmpl[pos] == '\t' || tmpl[pos] == '\n' || tmpl[pos] == '\r')) {
            pos++;
          }
        }
        continue;
      }
    }

    // accumulate text
    size_t textStart = pos;
    while (pos < len) {
      if (pos + 1 < len && tmpl[pos] == '{' && (tmpl[pos + 1] == '{' || tmpl[pos + 1] == '%')) {
        break;
      }
      pos++;
    }
    if (pos > textStart) {
      tokens.push_back({TokenType::TEXT, tmpl.substr(textStart, pos - textStart)});
    }
  }

  tokens.push_back({TokenType::END_OF_INPUT, ""});
  return tokens;
}

std::vector<Token> ChatTemplateEngine::tokenizeExpr(const std::string& content) {
  std::vector<Token> tokens;
  size_t pos = 0;
  size_t len = content.size();

  auto skipWs = [&]() {
    while (pos < len && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r'))
      pos++;
  };

  while (pos < len) {
    skipWs();
    if (pos >= len) break;

    char c = content[pos];

    // string literal
    if (c == '\'' || c == '"') {
      char quote = c;
      pos++;
      std::string s;
      while (pos < len && content[pos] != quote) {
        if (content[pos] == '\\' && pos + 1 < len) {
          pos++;
          switch (content[pos]) {
            case 'n':
              s += '\n';
              break;
            case 't':
              s += '\t';
              break;
            case 'r':
              s += '\r';
              break;
            case '\\':
              s += '\\';
              break;
            case '\'':
              s += '\'';
              break;
            case '"':
              s += '"';
              break;
            default:
              s += '\\';
              s += content[pos];
              break;
          }
        } else {
          s += content[pos];
        }
        pos++;
      }
      if (pos < len) pos++;  // skip closing quote
      tokens.push_back({TokenType::STRING_LIT, std::move(s)});
      continue;
    }

    // integer literal
    if (c >= '0' && c <= '9') {
      std::string num;
      while (pos < len && content[pos] >= '0' && content[pos] <= '9') {
        num += content[pos++];
      }
      tokens.push_back({TokenType::INT_LIT, std::move(num)});
      continue;
    }

    // two-char operators
    if (pos + 1 < len) {
      std::string twoChar = content.substr(pos, 2);
      if (twoChar == "==") {
        tokens.push_back({TokenType::EQ, "=="});
        pos += 2;
        continue;
      }
      if (twoChar == "!=") {
        tokens.push_back({TokenType::NEQ, "!="});
        pos += 2;
        continue;
      }
      if (twoChar == "<=") {
        tokens.push_back({TokenType::LTE, "<="});
        pos += 2;
        continue;
      }
      if (twoChar == ">=") {
        tokens.push_back({TokenType::GTE, ">="});
        pos += 2;
        continue;
      }
    }

    // single-char operators
    switch (c) {
      case '.':
        tokens.push_back({TokenType::DOT, "."});
        pos++;
        continue;
      case '[':
        tokens.push_back({TokenType::LBRACKET, "["});
        pos++;
        continue;
      case ']':
        tokens.push_back({TokenType::RBRACKET, "]"});
        pos++;
        continue;
      case '(':
        tokens.push_back({TokenType::LPAREN, "("});
        pos++;
        continue;
      case ')':
        tokens.push_back({TokenType::RPAREN, ")"});
        pos++;
        continue;
      case '|':
        tokens.push_back({TokenType::PIPE, "|"});
        pos++;
        continue;
      case ',':
        tokens.push_back({TokenType::COMMA, ","});
        pos++;
        continue;
      case '=':
        tokens.push_back({TokenType::ASSIGN, "="});
        pos++;
        continue;
      case '+':
        tokens.push_back({TokenType::PLUS, "+"});
        pos++;
        continue;
      case '-':
        tokens.push_back({TokenType::MINUS, "-"});
        pos++;
        continue;
      case '<':
        tokens.push_back({TokenType::LT, "<"});
        pos++;
        continue;
      case '>':
        tokens.push_back({TokenType::GT, ">"});
        pos++;
        continue;
      case '%':
        tokens.push_back({TokenType::MODULO, "%"});
        pos++;
        continue;
      case ':':
        tokens.push_back({TokenType::COLON, ":"});
        pos++;
        continue;
      case '~':
        tokens.push_back({TokenType::TILDE, "~"});
        pos++;
        continue;
      default:
        break;
    }

    // identifier or keyword
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
      std::string id;
      while (pos < len &&
             ((content[pos] >= 'a' && content[pos] <= 'z') || (content[pos] >= 'A' && content[pos] <= 'Z') ||
              (content[pos] >= '0' && content[pos] <= '9') || content[pos] == '_')) {
        id += content[pos++];
      }
      tokens.push_back({TokenType::IDENTIFIER, std::move(id)});
      continue;
    }

    // skip unknown characters
    LOGE("ChatTemplate: unknown char '%c' at pos %zu", c, pos);
    pos++;
  }

  return tokens;
}

class Parser {
 public:
  explicit Parser(const std::vector<Token>& tokens) : tokens_(tokens), pos_(0) {}

  std::vector<StmtPtr> parseAll() {
    std::vector<StmtPtr> stmts;
    while (!atEnd()) {
      if (auto stmt = parseStmt()) {
        stmts.push_back(std::move(stmt));
      }
    }
    return stmts;
  }

 private:
  const std::vector<Token>& tokens_;
  size_t pos_;

  const Token& peek() const { return tokens_[pos_]; }
  const Token& advance() { return tokens_[pos_++]; }
  bool atEnd() const { return pos_ >= tokens_.size() || tokens_[pos_].type == TokenType::END_OF_INPUT; }

  bool check(TokenType t) const { return !atEnd() && peek().type == t; }
  bool checkIdent(const std::string& name) const { return check(TokenType::IDENTIFIER) && peek().value == name; }

  bool match(TokenType t) {
    if (check(t)) {
      advance();
      return true;
    }
    return false;
  }

  void expect(TokenType t) {
    if (!match(t)) {
      LOGE("ChatTemplate: expected token type %d, got %d ('%s')", static_cast<int>(t), static_cast<int>(peek().type),
           peek().value.c_str());
    }
  }

  void expectIdent(const std::string& name) {
    if (!checkIdent(name)) {
      LOGE("ChatTemplate: expected '%s', got '%s'", name.c_str(), peek().value.c_str());
    }
    advance();
  }

  StmtPtr parseStmt() {
    if (check(TokenType::TEXT)) {
      return parseText();
    }
    if (check(TokenType::VAR_BEGIN)) {
      return parsePrint();
    }
    if (check(TokenType::BLOCK_BEGIN)) {
      return parseBlock();
    }
    // skip unexpected token
    advance();
    return nullptr;
  }

  StmtPtr parseText() {
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::TEXT;
    stmt->textContent = advance().value;
    return stmt;
  }

  StmtPtr parsePrint() {
    expect(TokenType::VAR_BEGIN);
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::PRINT;
    stmt->printExpr = parseExpr();
    expect(TokenType::VAR_END);
    return stmt;
  }

  StmtPtr parseBlock() {
    expect(TokenType::BLOCK_BEGIN);

    if (checkIdent("if")) {
      return parseIf();
    }
    if (checkIdent("for")) {
      return parseFor();
    }
    if (checkIdent("set")) {
      return parseSet();
    }

    // skip unknown block
    LOGE("ChatTemplate: unknown block keyword '%s'", peek().value.c_str());
    while (!atEnd() && !check(TokenType::BLOCK_END)) advance();
    match(TokenType::BLOCK_END);
    return nullptr;
  }

  StmtPtr parseIf() {
    expectIdent("if");
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::IF;

    // first branch: if condition
    Stmt::IfBranch branch;
    branch.condition = parseExpr();
    expect(TokenType::BLOCK_END);

    // parse body until elif/else/endif
    branch.body = parseUntilBlock({"elif", "else", "endif"});
    stmt->branches.push_back(std::move(branch));

    // handle elif / else
    while (true) {
      expect(TokenType::BLOCK_BEGIN);
      if (checkIdent("elif")) {
        advance();
        Stmt::IfBranch elifBranch;
        elifBranch.condition = parseExpr();
        expect(TokenType::BLOCK_END);
        elifBranch.body = parseUntilBlock({"elif", "else", "endif"});
        stmt->branches.push_back(std::move(elifBranch));
      } else if (checkIdent("else")) {
        advance();
        expect(TokenType::BLOCK_END);
        Stmt::IfBranch elseBranch;
        elseBranch.condition = nullptr;  // unconditional
        elseBranch.body = parseUntilBlock({"endif"});
        stmt->branches.push_back(std::move(elseBranch));
        expect(TokenType::BLOCK_BEGIN);
        expectIdent("endif");
        expect(TokenType::BLOCK_END);
        break;
      } else if (checkIdent("endif")) {
        advance();
        expect(TokenType::BLOCK_END);
        break;
      } else {
        LOGE("ChatTemplate: expected elif/else/endif, got '%s'", peek().value.c_str());
        break;
      }
    }

    return stmt;
  }

  StmtPtr parseFor() {
    expectIdent("for");
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::FOR;

    if (!check(TokenType::IDENTIFIER)) {
      LOGE("ChatTemplate: expected loop variable name");
      return nullptr;
    }
    stmt->loopVar = advance().value;
    expectIdent("in");
    stmt->iterExpr = parseExpr();
    expect(TokenType::BLOCK_END);

    stmt->forBody = parseUntilBlock({"endfor"});
    expect(TokenType::BLOCK_BEGIN);
    expectIdent("endfor");
    expect(TokenType::BLOCK_END);

    return stmt;
  }

  StmtPtr parseSet() {
    expectIdent("set");
    auto stmt = std::make_unique<Stmt>();
    stmt->kind = Stmt::SET;

    if (!check(TokenType::IDENTIFIER)) {
      LOGE("ChatTemplate: expected variable name after set");
      return nullptr;
    }
    stmt->setVar = advance().value;

    // handle dotted name: set ns.member = expr
    while (check(TokenType::DOT)) {
      advance();
      if (!check(TokenType::IDENTIFIER)) {
        LOGE("ChatTemplate: expected member name after '.'");
        break;
      }
      stmt->setVar += "." + advance().value;
    }

    expect(TokenType::ASSIGN);
    stmt->setExpr = parseExpr();
    expect(TokenType::BLOCK_END);

    return stmt;
  }

  // parse statements until we see a block with one of the given keywords
  std::vector<StmtPtr> parseUntilBlock(const std::vector<std::string>& endKeywords) {
    std::vector<StmtPtr> stmts;
    while (!atEnd()) {
      // peek ahead: is this a {% endKeyword %} ?
      if (check(TokenType::BLOCK_BEGIN)) {
        size_t savedPos = pos_;
        advance();  // skip {%
        bool isEnd = false;
        for (auto& kw : endKeywords) {
          if (checkIdent(kw)) {
            isEnd = true;
            break;
          }
        }
        pos_ = savedPos;  // restore
        if (isEnd) break;
      }

      if (auto stmt = parseStmt()) {
        stmts.push_back(std::move(stmt));
      }
    }
    return stmts;
  }

  ExprPtr parseExpr() { return parseOr(); }

  ExprPtr parseOr() {
    auto left = parseAnd();
    while (checkIdent("or")) {
      advance();
      auto right = parseAnd();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::BINARY_OP;
      node->op = "or";
      node->left = std::move(left);
      node->right = std::move(right);
      left = std::move(node);
    }
    return left;
  }

  ExprPtr parseAnd() {
    auto left = parseNot();
    while (checkIdent("and")) {
      advance();
      auto right = parseNot();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::BINARY_OP;
      node->op = "and";
      node->left = std::move(left);
      node->right = std::move(right);
      left = std::move(node);
    }
    return left;
  }

  ExprPtr parseNot() {
    if (checkIdent("not")) {
      advance();
      auto operand = parseNot();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::UNARY_OP;
      node->op = "not";
      node->left = std::move(operand);
      return node;
    }
    return parseComparison();
  }

  ExprPtr parseComparison() {
    auto left = parseAddConcat();
    while (check(TokenType::EQ) || check(TokenType::NEQ) || check(TokenType::LT) || check(TokenType::GT) ||
           check(TokenType::LTE) || check(TokenType::GTE) || checkIdent("is") || checkIdent("in") ||
           checkIdent("not")) {
      // handle "is defined", "is not defined", "not in"
      if (checkIdent("is")) {
        advance();
        bool negated = false;
        if (checkIdent("not")) {
          advance();
          negated = true;
        }
        std::string test = "defined";
        if (check(TokenType::IDENTIFIER)) {
          test = advance().value;
        }
        auto node = std::make_unique<Expr>();
        node->kind = Expr::BINARY_OP;
        node->op = negated ? "is not " + test : "is " + test;
        node->left = std::move(left);
        return node;
      }
      // handle "not in"
      if (checkIdent("not")) {
        size_t savedPos = pos_;
        advance();
        if (checkIdent("in")) {
          advance();
          auto right = parseAddConcat();
          auto node = std::make_unique<Expr>();
          node->kind = Expr::BINARY_OP;
          node->op = "not in";
          node->left = std::move(left);
          node->right = std::move(right);
          return node;
        }
        pos_ = savedPos;
        break;
      }
      if (checkIdent("in")) {
        advance();
        auto right = parseAddConcat();
        auto node = std::make_unique<Expr>();
        node->kind = Expr::BINARY_OP;
        node->op = "in";
        node->left = std::move(left);
        node->right = std::move(right);
        return node;
      }

      std::string op = advance().value;
      auto right = parseAddConcat();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::BINARY_OP;
      node->op = op;
      node->left = std::move(left);
      node->right = std::move(right);
      left = std::move(node);
    }
    return left;
  }

  ExprPtr parseAddConcat() {
    auto left = parseModulo();
    while (check(TokenType::PLUS) || check(TokenType::MINUS) || check(TokenType::TILDE)) {
      std::string op = advance().value;
      auto right = parseModulo();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::BINARY_OP;
      node->op = op;
      node->left = std::move(left);
      node->right = std::move(right);
      left = std::move(node);
    }
    return left;
  }

  ExprPtr parseModulo() {
    auto left = parseFilter();
    while (check(TokenType::MODULO)) {
      advance();
      auto right = parseFilter();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::BINARY_OP;
      node->op = "%";
      node->left = std::move(left);
      node->right = std::move(right);
      left = std::move(node);
    }
    return left;
  }

  ExprPtr parseFilter() {
    auto expr = parsePostfix();
    while (check(TokenType::PIPE)) {
      advance();
      if (!check(TokenType::IDENTIFIER)) {
        LOGE("ChatTemplate: expected filter name");
        break;
      }
      std::string filterName = advance().value;
      auto node = std::make_unique<Expr>();
      node->kind = Expr::FILTER;
      node->op = filterName;
      node->left = std::move(expr);
      // filter arguments: |default("val")
      if (check(TokenType::LPAREN)) {
        advance();
        while (!check(TokenType::RPAREN) && !atEnd()) {
          node->args.push_back(parseExpr());
          if (!check(TokenType::RPAREN)) match(TokenType::COMMA);
        }
        expect(TokenType::RPAREN);
      }
      expr = std::move(node);
    }
    return expr;
  }

  ExprPtr parsePostfix() {
    auto expr = parsePrimary();

    while (true) {
      if (check(TokenType::DOT)) {
        advance();
        if (!check(TokenType::IDENTIFIER)) {
          LOGE("ChatTemplate: expected member name after '.'");
          break;
        }
        std::string member = advance().value;

        // check if it's a method call: expr.method(...)
        if (check(TokenType::LPAREN)) {
          advance();
          auto node = std::make_unique<Expr>();
          node->kind = Expr::METHOD_CALL;
          node->op = member;
          node->left = std::move(expr);
          while (!check(TokenType::RPAREN) && !atEnd()) {
            // check for keyword argument: name=value
            if (check(TokenType::IDENTIFIER) && pos_ + 1 < tokens_.size() &&
                tokens_[pos_ + 1].type == TokenType::ASSIGN) {
              std::string argName = advance().value;
              advance();  // skip =
              node->kwargs.emplace_back(argName, parseExpr());
            } else {
              node->args.push_back(parseExpr());
            }
            if (!check(TokenType::RPAREN)) match(TokenType::COMMA);
          }
          expect(TokenType::RPAREN);
          expr = std::move(node);
        } else {
          auto node = std::make_unique<Expr>();
          node->kind = Expr::MEMBER_ACCESS;
          node->op = member;
          node->left = std::move(expr);
          expr = std::move(node);
        }
      } else if (check(TokenType::LBRACKET)) {
        advance();
        // Check if this is a slice expression: [start:stop] or [start:stop:step]
        // Slice forms: [:], [start:], [:stop], [::step], [start:stop:step], etc.
        bool isSlice = false;

        // Parse first element (start) or detect leading colon
        ExprPtr first;
        if (check(TokenType::COLON)) {
          isSlice = true;
          // no start expr
        } else if (!check(TokenType::RBRACKET)) {
          first = parseExpr();
          if (check(TokenType::COLON)) {
            isSlice = true;
          }
        }

        if (isSlice) {
          // Build slice node: sliceArgs = [start, stop, step] (nullptr = omitted)
          auto node = std::make_unique<Expr>();
          node->kind = Expr::SLICE_ACCESS;
          node->left = std::move(expr);
          node->sliceArgs.resize(3);              // [start, stop, step]
          node->sliceArgs[0] = std::move(first);  // may be nullptr

          // consume first ':'
          expect(TokenType::COLON);

          // parse stop (if not ':' or ']')
          if (!check(TokenType::COLON) && !check(TokenType::RBRACKET)) {
            node->sliceArgs[1] = parseExpr();
          }

          // optional second ':' for step
          if (match(TokenType::COLON)) {
            if (!check(TokenType::RBRACKET)) {
              node->sliceArgs[2] = parseExpr();
            }
          }

          expect(TokenType::RBRACKET);
          expr = std::move(node);
        } else {
          // Simple index access
          auto node = std::make_unique<Expr>();
          node->kind = Expr::INDEX_ACCESS;
          node->left = std::move(expr);
          node->right = std::move(first);
          expect(TokenType::RBRACKET);
          expr = std::move(node);
        }
      } else {
        break;
      }
    }
    return expr;
  }

  ExprPtr parsePrimary() {
    // string literal
    if (check(TokenType::STRING_LIT)) {
      auto node = std::make_unique<Expr>();
      node->kind = Expr::STRING_LITERAL;
      node->strValue = advance().value;
      return node;
    }

    // unary minus (negative integer)
    if (check(TokenType::MINUS)) {
      advance();
      if (check(TokenType::INT_LIT)) {
        auto node = std::make_unique<Expr>();
        node->kind = Expr::INT_LITERAL;
        node->intValue = -std::stoll(advance().value);
        return node;
      }
      // unary minus on expression
      auto operand = parsePrimary();
      auto node = std::make_unique<Expr>();
      node->kind = Expr::UNARY_OP;
      node->op = "-";
      node->left = std::move(operand);
      return node;
    }

    // integer literal
    if (check(TokenType::INT_LIT)) {
      auto node = std::make_unique<Expr>();
      node->kind = Expr::INT_LITERAL;
      node->intValue = std::stoll(advance().value);
      return node;
    }

    // parenthesized expression
    if (check(TokenType::LPAREN)) {
      advance();
      auto expr = parseExpr();
      expect(TokenType::RPAREN);
      return expr;
    }

    // identifier / keyword / function call
    if (check(TokenType::IDENTIFIER)) {
      std::string name = advance().value;

      // boolean literals
      if (name == "true" || name == "True") {
        auto node = std::make_unique<Expr>();
        node->kind = Expr::BOOL_LITERAL;
        node->boolValue = true;
        return node;
      }
      if (name == "false" || name == "False") {
        auto node = std::make_unique<Expr>();
        node->kind = Expr::BOOL_LITERAL;
        node->boolValue = false;
        return node;
      }
      if (name == "none" || name == "None") {
        auto node = std::make_unique<Expr>();
        node->kind = Expr::NONE_LITERAL;
        return node;
      }

      // function call: name(...)
      if (check(TokenType::LPAREN)) {
        advance();
        auto node = std::make_unique<Expr>();
        node->kind = Expr::FUNC_CALL;
        node->strValue = name;
        while (!check(TokenType::RPAREN) && !atEnd()) {
          // check for keyword argument: name=value
          if (check(TokenType::IDENTIFIER) && pos_ + 1 < tokens_.size() &&
              tokens_[pos_ + 1].type == TokenType::ASSIGN) {
            std::string argName = advance().value;
            advance();  // skip =
            node->kwargs.emplace_back(argName, parseExpr());
          } else {
            node->args.push_back(parseExpr());
          }
          if (!check(TokenType::RPAREN)) match(TokenType::COMMA);
        }
        expect(TokenType::RPAREN);
        return node;
      }

      auto node = std::make_unique<Expr>();
      node->kind = Expr::IDENTIFIER;
      node->strValue = name;
      return node;
    }

    LOGE("ChatTemplate: unexpected token '%s'", peek().value.c_str());
    auto node = std::make_unique<Expr>();
    node->kind = Expr::NONE_LITERAL;
    if (!atEnd()) advance();
    return node;
  }
};

std::vector<StmtPtr> ChatTemplateEngine::parse(const std::vector<Token>& tokens) {
  Parser parser(tokens);
  return parser.parseAll();
}

static Value getMember(const Value& val, const std::string& member) {
  if (val.isMessage()) {
    const auto& msg = val.asMessage();
    if (member == "role") return Value(msg.role);
    if (member == "content") return Value(msg.content);
  }
  if (val.isMessageList()) {
    if (member == "length" || member == "size") {
      return Value(static_cast<int64_t>(val.asMessageList().size()));
    }
  }
  if (val.isString()) {
    if (member == "length" || member == "size") {
      return Value(static_cast<int64_t>(val.asString().size()));
    }
  }
  return {};
}

static void computeSliceIndices(int64_t len, const Value& startVal, const Value& stopVal, const Value& stepVal,
                                int64_t& start, int64_t& stop, int64_t& step) {
  step = stepVal.isInt() ? stepVal.asInt() : 1;
  if (step == 0) step = 1;  // avoid infinite loop

  if (step > 0) {
    start = startVal.isInt() ? startVal.asInt() : 0;
    stop = stopVal.isInt() ? stopVal.asInt() : len;
  } else {
    start = startVal.isInt() ? startVal.asInt() : len - 1;
    stop = stopVal.isInt() ? stopVal.asInt() : -(len + 1);
  }

  // normalize negative indices
  if (start < 0) start += len;
  if (stop < 0) stop += len;

  // clamp
  if (step > 0) {
    if (start < 0) start = 0;
    if (start > len) start = len;
    if (stop < 0) stop = 0;
    if (stop > len) stop = len;
  } else {
    if (start < -1) start = -1;
    if (start >= len) start = len - 1;
    if (stop < -1) stop = -1;
    if (stop >= len) stop = len - 1;
  }
}

template <typename T>
static std::vector<T> sliceVector(const std::vector<T>& vec, const Value& startVal, const Value& stopVal,
                                  const Value& stepVal) {
  int64_t start, stop, step;
  computeSliceIndices(static_cast<int64_t>(vec.size()), startVal, stopVal, stepVal, start, stop, step);

  std::vector<T> result;
  if (step > 0) {
    for (int64_t i = start; i < stop; i += step) {
      result.push_back(vec[i]);
    }
  } else {
    for (int64_t i = start; i > stop; i += step) {
      result.push_back(vec[i]);
    }
  }
  return result;
}

static Value sliceValue(const Value& val, const Value& startVal, const Value& stopVal, const Value& stepVal) {
  if (val.isMessageList()) {
    return Value(sliceVector(val.asMessageList(), startVal, stopVal, stepVal));
  }
  if (val.isStringList()) {
    return Value(sliceVector(val.asStringList(), startVal, stopVal, stepVal));
  }
  if (val.isString()) {
    const auto& s = val.asString();
    int64_t start, stop, step;
    computeSliceIndices(static_cast<int64_t>(s.size()), startVal, stopVal, stepVal, start, stop, step);
    std::string result;
    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) {
        result += s[i];
      }
    } else {
      for (int64_t i = start; i > stop; i += step) {
        result += s[i];
      }
    }
    return Value(std::move(result));
  }
  return {};
}

static Value getIndex(const Value& val, const Value& index) {
  if (val.isMessage() && index.isString()) {
    return getMember(val, index.asString());
  }
  if (val.isMessageList() && index.isInt()) {
    auto idx = index.asInt();
    const auto& list = val.asMessageList();
    if (idx < 0) idx += static_cast<int64_t>(list.size());
    if (idx >= 0 && idx < static_cast<int64_t>(list.size())) {
      return Value(list[idx]);
    }
  }
  if (val.isStringList() && index.isInt()) {
    auto idx = index.asInt();
    const auto& list = val.asStringList();
    if (idx < 0) idx += static_cast<int64_t>(list.size());
    if (idx >= 0 && idx < static_cast<int64_t>(list.size())) {
      return Value(list[idx]);
    }
  }
  if (val.isString() && index.isInt()) {
    auto idx = index.asInt();
    const auto& s = val.asString();
    if (idx < 0) idx += static_cast<int64_t>(s.size());
    if (idx >= 0 && idx < static_cast<int64_t>(s.size())) {
      return Value(std::string(1, s[idx]));
    }
  }
  return {};
}

static std::string applyFilter(const Value& val, const std::string& filterName, const std::vector<Value>& args) {
  if (filterName == "trim") {
    return trimWhitespace(val.toString());
  }
  if (filterName == "length" || filterName == "count") {
    if (val.isString()) return std::to_string(val.asString().size());
    if (val.isMessageList()) return std::to_string(val.asMessageList().size());
    return "0";
  }
  if (filterName == "upper") {
    std::string s = val.toString();
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
  }
  if (filterName == "lower") {
    std::string s = val.toString();
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
  }
  if (filterName == "default" || filterName == "d") {
    if (val.isNone() || (val.isString() && val.asString().empty())) {
      return args.empty() ? "" : args[0].toString();
    }
    return val.toString();
  }
  if (filterName == "first") {
    if (val.isMessageList() && !val.asMessageList().empty()) {
      return "[ChatMessage]";
    }
    if (val.isString() && !val.asString().empty()) {
      return {1, val.asString()[0]};
    }
    return "";
  }
  if (filterName == "last") {
    if (val.isString() && !val.asString().empty()) {
      return {1, val.asString().back()};
    }
    return "";
  }
  if (filterName == "string") {
    return val.toString();
  }
  if (filterName == "int") {
    if (val.isInt()) return std::to_string(val.asInt());
    if (val.isString()) {
      // try parse
      return val.asString();
    }
    return "0";
  }

  LOGE("ChatTemplate: unknown filter '%s'", filterName.c_str());
  return val.toString();
}

Value ChatTemplateEngine::evalExpr(const Expr& expr, EvalContext& ctx) {
  switch (expr.kind) {
    case Expr::STRING_LITERAL:
      return Value(expr.strValue);

    case Expr::BOOL_LITERAL:
      return Value(expr.boolValue);

    case Expr::INT_LITERAL:
      return Value(expr.intValue);

    case Expr::NONE_LITERAL:
      return {};

    case Expr::IDENTIFIER:
      return ctx.get(expr.strValue);

    case Expr::MEMBER_ACCESS: {
      auto obj = evalExpr(*expr.left, ctx);
      // first try flat key lookup for namespace support: "ns.member"
      if (expr.left->kind == Expr::IDENTIFIER) {
        auto flatKey = expr.left->strValue + "." + expr.op;
        auto flatVal = ctx.get(flatKey);
        if (!flatVal.isNone()) return flatVal;
      }
      return getMember(obj, expr.op);
    }

    case Expr::INDEX_ACCESS: {
      auto obj = evalExpr(*expr.left, ctx);
      auto idx = evalExpr(*expr.right, ctx);
      return getIndex(obj, idx);
    }

    case Expr::SLICE_ACCESS: {
      auto obj = evalExpr(*expr.left, ctx);
      Value startVal = expr.sliceArgs[0] ? evalExpr(*expr.sliceArgs[0], ctx) : Value();
      Value stopVal = expr.sliceArgs[1] ? evalExpr(*expr.sliceArgs[1], ctx) : Value();
      Value stepVal = expr.sliceArgs[2] ? evalExpr(*expr.sliceArgs[2], ctx) : Value();
      return sliceValue(obj, startVal, stopVal, stepVal);
    }

    case Expr::BINARY_OP: {
      // short-circuit for and/or
      if (expr.op == "and") {
        auto left = evalExpr(*expr.left, ctx);
        if (!left.truthy()) return Value(false);
        auto right = evalExpr(*expr.right, ctx);
        return Value(right.truthy());
      }
      if (expr.op == "or") {
        auto left = evalExpr(*expr.left, ctx);
        if (left.truthy()) return Value(true);
        auto right = evalExpr(*expr.right, ctx);
        return Value(right.truthy());
      }
      if (expr.op == "==" || expr.op == "!=") {
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        bool eq = false;
        if (left.isString() && right.isString()) {
          eq = left.asString() == right.asString();
        } else if (left.isBool() && right.isBool()) {
          eq = left.asBool() == right.asBool();
        } else if (left.isInt() && right.isInt()) {
          eq = left.asInt() == right.asInt();
        } else if (left.isNone() && right.isNone()) {
          eq = true;
        } else if (left.isNone() || right.isNone()) {
          eq = false;
        }
        return Value(expr.op == "==" ? eq : !eq);
      }
      if (expr.op == "<" || expr.op == ">" || expr.op == "<=" || expr.op == ">=") {
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        if (left.isInt() && right.isInt()) {
          int64_t l = left.asInt(), r = right.asInt();
          if (expr.op == "<") return Value(l < r);
          if (expr.op == ">") return Value(l > r);
          if (expr.op == "<=") return Value(l <= r);
          if (expr.op == ">=") return Value(l >= r);
        }
        if (left.isString() && right.isString()) {
          int cmp = left.asString().compare(right.asString());
          if (expr.op == "<") return Value(cmp < 0);
          if (expr.op == ">") return Value(cmp > 0);
          if (expr.op == "<=") return Value(cmp <= 0);
          if (expr.op == ">=") return Value(cmp >= 0);
        }
        return Value(false);
      }
      if (expr.op == "+") {
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        if (left.isString() && right.isString()) {
          return Value(left.asString() + right.asString());
        }
        if (left.isInt() && right.isInt()) {
          return Value(left.asInt() + right.asInt());
        }
        return Value(left.toString() + right.toString());
      }
      if (expr.op == "-") {
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        if (left.isInt() && right.isInt()) {
          return Value(left.asInt() - right.asInt());
        }
        return Value(static_cast<int64_t>(0));
      }
      if (expr.op == "~") {
        // string concatenation
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        return Value(left.toString() + right.toString());
      }
      if (expr.op == "%") {
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        if (left.isInt() && right.isInt() && right.asInt() != 0) {
          return Value(left.asInt() % right.asInt());
        }
        return Value(static_cast<int64_t>(0));
      }
      // "is defined" / "is not defined" / "is none" / "is not none" / "is string" etc.
      if (expr.op == "is defined") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isNone());
      }
      if (expr.op == "is not defined") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isNone());
      }
      if (expr.op == "is none") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isNone());
      }
      if (expr.op == "is not none") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isNone());
      }
      if (expr.op == "is string") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isString());
      }
      if (expr.op == "is not string") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isString());
      }
      if (expr.op == "is false") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isBool() && !val.asBool());
      }
      if (expr.op == "is not false") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isBool() || val.asBool());
      }
      if (expr.op == "is true") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isBool() && val.asBool());
      }
      if (expr.op == "is not true") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isBool() || !val.asBool());
      }
      if (expr.op == "is boolean") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isBool());
      }
      if (expr.op == "is not boolean") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isBool());
      }
      if (expr.op == "is integer") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(val.isInt());
      }
      if (expr.op == "is not integer") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.isInt());
      }
      // "in" / "not in"
      if (expr.op == "in" || expr.op == "not in") {
        auto left = evalExpr(*expr.left, ctx);
        auto right = evalExpr(*expr.right, ctx);
        bool found = false;
        if (right.isString() && left.isString()) {
          found = right.asString().find(left.asString()) != std::string::npos;
        }
        return Value(expr.op == "in" ? found : !found);
      }

      LOGE("ChatTemplate: unknown binary op '%s'", expr.op.c_str());
      return {};
    }

    case Expr::UNARY_OP: {
      if (expr.op == "not") {
        auto val = evalExpr(*expr.left, ctx);
        return Value(!val.truthy());
      }
      if (expr.op == "-") {
        auto val = evalExpr(*expr.left, ctx);
        if (val.isInt()) return Value(-val.asInt());
        return Value(static_cast<int64_t>(0));
      }
      return {};
    }

    case Expr::FILTER: {
      auto val = evalExpr(*expr.left, ctx);
      // evaluate filter arguments
      std::vector<Value> filterArgs;
      for (auto& arg : expr.args) {
        filterArgs.push_back(evalExpr(*arg, ctx));
      }
      // special handling: filter returns Value for some filters
      if (expr.op == "length" || expr.op == "count") {
        if (val.isString()) return Value(static_cast<int64_t>(val.asString().size()));
        if (val.isMessageList()) return Value(static_cast<int64_t>(val.asMessageList().size()));
        return Value(static_cast<int64_t>(0));
      }
      if (expr.op == "default" || expr.op == "d") {
        if (val.isNone() || (val.isString() && val.asString().empty())) {
          return filterArgs.empty() ? Value(std::string("")) : filterArgs[0];
        }
        return val;
      }
      if (expr.op == "first") {
        if (val.isMessageList() && !val.asMessageList().empty()) {
          return Value(val.asMessageList().front());
        }
        if (val.isString() && !val.asString().empty()) {
          return Value(std::string(1, val.asString()[0]));
        }
        return {};
      }
      if (expr.op == "last") {
        if (val.isMessageList() && !val.asMessageList().empty()) {
          return Value(val.asMessageList().back());
        }
        if (val.isString() && !val.asString().empty()) {
          return Value(std::string(1, val.asString().back()));
        }
        return {};
      }
      // for other filters, return string result
      return Value(applyFilter(val, expr.op, filterArgs));
    }

    case Expr::FUNC_CALL: {
      if (expr.strValue == "raise_exception") {
        std::string msg = "ChatTemplate error";
        if (!expr.args.empty()) {
          msg = evalExpr(*expr.args[0], ctx).toString();
        }
        LOGE("ChatTemplate raise_exception: %s", msg.c_str());
        return {};
      }
      if (expr.strValue == "namespace") {
        // namespace() is handled specially in SET statement
        // if called outside SET, just return a marker value
        return Value(true);
      }
      if (expr.strValue == "range") {
        // range(n) → generate n items (not fully supported, return none)
        return {};
      }
      if (expr.strValue == "strftime_now") {
        // not supported, return empty
        return Value(std::string(""));
      }
      LOGE("ChatTemplate: unknown function '%s'", expr.strValue.c_str());
      return {};
    }

    case Expr::METHOD_CALL: {
      auto obj = evalExpr(*expr.left, ctx);
      if (expr.op == "strip" || expr.op == "trim") {
        std::string chars = "\n\r\t ";
        if (!expr.args.empty()) {
          chars = evalExpr(*expr.args[0], ctx).toString();
        }
        std::string s = obj.toString();
        size_t start = s.find_first_not_of(chars);
        size_t end = s.find_last_not_of(chars);
        if (start == std::string::npos) return Value(std::string(""));
        return Value(s.substr(start, end - start + 1));
      }
      if (expr.op == "lstrip") {
        std::string chars = "\n\r\t ";
        if (!expr.args.empty()) {
          chars = evalExpr(*expr.args[0], ctx).toString();
        }
        std::string s = obj.toString();
        size_t start = s.find_first_not_of(chars);
        if (start == std::string::npos) return Value(std::string(""));
        return Value(s.substr(start));
      }
      if (expr.op == "rstrip") {
        std::string chars = "\n\r\t ";
        if (!expr.args.empty()) {
          chars = evalExpr(*expr.args[0], ctx).toString();
        }
        std::string s = obj.toString();
        size_t end = s.find_last_not_of(chars);
        if (end == std::string::npos) return Value(std::string(""));
        return Value(s.substr(0, end + 1));
      }
      if (expr.op == "split") {
        std::string sep = " ";
        if (!expr.args.empty()) {
          sep = evalExpr(*expr.args[0], ctx).toString();
        }
        std::string s = obj.toString();
        std::vector<std::string> parts;
        if (sep.empty()) {
          parts.push_back(s);
        } else {
          size_t start = 0;
          while (true) {
            size_t found = s.find(sep, start);
            if (found == std::string::npos) {
              parts.push_back(s.substr(start));
              break;
            }
            parts.push_back(s.substr(start, found - start));
            start = found + sep.size();
          }
        }
        return Value(std::move(parts));
      }
      if (expr.op == "startswith") {
        if (!expr.args.empty()) {
          std::string prefix = evalExpr(*expr.args[0], ctx).toString();
          std::string s = obj.toString();
          bool result = s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
          return Value(result);
        }
        return Value(false);
      }
      if (expr.op == "endswith") {
        if (!expr.args.empty()) {
          std::string suffix = evalExpr(*expr.args[0], ctx).toString();
          std::string s = obj.toString();
          bool result = s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
          return Value(result);
        }
        return Value(false);
      }
      if (expr.op == "upper") {
        std::string s = obj.toString();
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
        return Value(std::move(s));
      }
      if (expr.op == "lower") {
        std::string s = obj.toString();
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return Value(std::move(s));
      }
      LOGE("ChatTemplate: unknown method '%s'", expr.op.c_str());
      return {};
    }
  }
  return {};
}

void ChatTemplateEngine::evalStmt(const Stmt& stmt, EvalContext& ctx, std::string& output) {
  switch (stmt.kind) {
    case Stmt::TEXT:
      output += stmt.textContent;
      break;

    case Stmt::PRINT: {
      auto val = evalExpr(*stmt.printExpr, ctx);
      output += val.toString();
      break;
    }

    case Stmt::IF: {
      for (auto& branch : stmt.branches) {
        if (!branch.condition || evalExpr(*branch.condition, ctx).truthy()) {
          evalStmts(branch.body, ctx, output);
          break;
        }
      }
      break;
    }

    case Stmt::FOR: {
      auto iterVal = evalExpr(*stmt.iterExpr, ctx);

      // helper lambda to set loop variables and execute body
      auto runLoop = [&](size_t i, size_t total) {
        ctx.set("loop.index", Value(static_cast<int64_t>(i + 1)));
        ctx.set("loop.index0", Value(static_cast<int64_t>(i)));
        ctx.set("loop.first", Value(i == 0));
        ctx.set("loop.last", Value(i == total - 1));
        ctx.set("loop.length", Value(static_cast<int64_t>(total)));
        ctx.set("loop.revindex", Value(static_cast<int64_t>(total - i)));
        ctx.set("loop.revindex0", Value(static_cast<int64_t>(total - i - 1)));
        evalStmts(stmt.forBody, ctx, output);
      };

      if (iterVal.isMessageList()) {
        const auto& list = iterVal.asMessageList();
        ctx.pushScope();
        for (size_t i = 0; i < list.size(); i++) {
          ctx.set(stmt.loopVar, Value(list[i]));
          runLoop(i, list.size());
        }
        ctx.popScope();
      } else if (iterVal.isStringList()) {
        const auto& list = iterVal.asStringList();
        ctx.pushScope();
        for (size_t i = 0; i < list.size(); i++) {
          ctx.set(stmt.loopVar, Value(list[i]));
          runLoop(i, list.size());
        }
        ctx.popScope();
      }
      break;
    }

    case Stmt::SET: {
      // special handling for namespace: {% set ns = namespace(key=val, ...) %}
      if (stmt.setExpr && stmt.setExpr->kind == Expr::FUNC_CALL && stmt.setExpr->strValue == "namespace") {
        ctx.set(stmt.setVar, Value(true));  // mark as namespace object
        for (auto& [key, valExpr] : stmt.setExpr->kwargs) {
          ctx.set(stmt.setVar + "." + key, evalExpr(*valExpr, ctx));
        }
      } else {
        auto val = evalExpr(*stmt.setExpr, ctx);
        ctx.set(stmt.setVar, std::move(val));
      }
      break;
    }
  }
}

void ChatTemplateEngine::evalStmts(const std::vector<StmtPtr>& stmts, EvalContext& ctx, std::string& output) {
  for (auto& stmt : stmts) {
    if (stmt) evalStmt(*stmt, ctx, output);
  }
}

std::string ChatTemplateEngine::render(const std::string& tmpl, const std::vector<ChatMessage>& messages,
                                       bool addGenerationPrompt, const std::string& bosToken,
                                       const std::string& eosToken) {
  // strip trailing whitespace from template (HuggingFace convention)
  std::string cleanTmpl = tmpl;
  while (!cleanTmpl.empty() && (cleanTmpl.back() == '\n' || cleanTmpl.back() == '\r' || cleanTmpl.back() == ' ')) {
    cleanTmpl.pop_back();
  }

  // lexer
  auto tokens = tokenize(cleanTmpl);

  // parser
  auto stmts = parse(tokens);

  // setup context
  EvalContext ctx;
  ctx.pushScope();
  ctx.set("messages", Value(messages));
  ctx.set("add_generation_prompt", Value(addGenerationPrompt));
  ctx.set("bos_token", Value(bosToken));
  ctx.set("eos_token", Value(eosToken));

  // evaluate
  std::string output;
  output.reserve(tmpl.size());
  evalStmts(stmts, ctx, output);

  return output;
}

std::string applyChatTemplate(const std::string& tmpl, const std::vector<ChatMessage>& messages,
                              bool addGenerationPrompt, const std::string& bosToken, const std::string& eosToken) {
  return ChatTemplateEngine::render(tmpl, messages, addGenerationPrompt, bosToken, eosToken);
}

}  // namespace tinygpt::tokenizer
