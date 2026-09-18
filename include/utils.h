// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 12/17/25.
//

#ifndef TESSERA_UTILS_H
#define TESSERA_UTILS_H

#include <unordered_map>
#include <clocale>
#include <cstring>
#include <cstdlib>
#include <string>

/// A uniform draw from [min, max), from a thread-local generator seeded once per thread.
inline double random_uniform(double min = -1.0, double max = 1.0) {
  thread_local std::mt19937 gen{std::random_device{}()};
  std::uniform_real_distribution<double> dist(min, max);
  return dist(gen);
}

/// True if LC_ALL, LC_CTYPE or LANG names a UTF-8 encoding.
inline bool envClaimsUTF8() {
  const char* vars[] = {"LC_ALL", "LC_CTYPE", "LANG"};
  for (auto v : vars) {
    const char* val = std::getenv(v);
    if (val && std::strstr(val, "UTF-8")) return true;
  }
  return false;
}

/// True if the C library's current LC_CTYPE locale names a UTF-8 encoding.
inline bool localClaimsUTF8() {
  const char* loc = std::setlocale(LC_CTYPE, "");
  return loc && std::strstr(loc, "UTF-8");
}

inline const std::unordered_map<std::string, std::string> subscriptLookup = {
  {"0", "₀"}, {"1", "₁"}, {"2", "₂"}, {"3", "₃"}, {"4", "₄"},
  {"5", "₅"}, {"6", "₆"}, {"7", "₇"}, {"8", "₈"}, {"9", "₉"},
  {"a", "ₐ"}, {"e", "ₑ"}, {"h", "ₕ"}, {"i", "ᵢ"}, {"j", "ⱼ"},
  {"k", "ₖ"}, {"l", "ₗ"}, {"m", "ₘ"}, {"n", "ₙ"}, {"o", "ₒ"},
  {"p", "ₚ"}, {"r", "ᵣ"}, {"s", "ₛ"}, {"t", "ₜ"}, {"u", "ᵤ"},
  {"v", "ᵥ"}, {"x", "ₓ"}, {"+", "₊"}, {"-", "₋"}, {"=", "₌"},
  {"(", "₍"}, {")", "₎"}, {",", "︐"}, {".", "․"}
};

inline const std::unordered_map<std::string, std::string> superscriptLookup = {
  {"0", "⁰"}, {"1", "¹"}, {"2", "²"}, {"3", "³"}, {"4", "⁴"},
  {"5", "⁵"}, {"6", "⁶"}, {"7", "⁷"}, {"8", "⁸"}, {"9", "⁹"},
  {"a", "ᵃ"}, {"b", "ᵇ"}, {"c", "ᶜ"}, {"d", "ᵈ"}, {"e", "ᵉ"},
  {"f", "ᶠ"}, {"g", "ᵍ"}, {"h", "ʰ"}, {"i", "ⁱ"}, {"j", "ʲ"},
  {"k", "ᵏ"}, {"l", "ˡ"}, {"m", "ᵐ"}, {"n", "ⁿ"}, {"o", "ᵒ"},
  {"p", "ᵖ"}, {"r", "ʳ"}, {"s", "ˢ"}, {"t", "ᵗ"}, {"u", "ᵘ"},
  {"v", "ᵛ"}, {"w", "ʷ"}, {"x", "ˣ"}, {"y", "ʸ"}, {"z", "ᶻ"},
  {"A", "ᴬ"}, {"B", "ᴮ"}, {"D", "ᴰ"}, {"E", "ᴱ"}, {"G", "ᴳ"},
  {"H", "ᴴ"}, {"I", "ᴵ"}, {"J", "ᴶ"}, {"K", "ᴷ"}, {"L", "ᴸ"},
  {"M", "ᴹ"}, {"N", "ᴺ"}, {"O", "ᴼ"}, {"P", "ᴾ"}, {"R", "ᴿ"},
  {"T", "ᵀ"}, {"U", "ᵁ"}, {"V", "ⱽ"}, {"W", "ᵂ"},
  {"+", "⁺"}, {"-", "⁻"}, {"=", "⁼"}, {"(", "⁽"}, {")", "⁾"},
  {"*", "﹡"}, {":", "ː"}, {";", "⸵"},{",", "︐"}, {"*", "﹡"},
  {"!", "ꜝ"}, {"?", "ˀ"}, {"/", "ᐟ"}
};

inline const std::unordered_map<std::string, std::string> greekLookup = {
  // Lowercase Greek letters
  {"\\alpha", "α"}, {"\\beta", "β"}, {"\\gamma", "γ"}, {"\\delta", "δ"},
  {"\\epsilon", "ε"}, {"\\zeta", "ζ"}, {"\\eta", "η"}, {"\\theta", "θ"},
  {"\\iota", "ι"}, {"\\kappa", "κ"}, {"\\lambda", "λ"}, {"\\mu", "μ"},
  {"\\nu", "ν"}, {"\\xi", "ξ"}, {"\\omicron", "ο"}, {"\\pi", "π"},
  {"\\rho", "ρ"}, {"\\sigma", "σ"}, {"\\tau", "τ"}, {"\\upsilon", "υ"},
  {"\\phi", "φ"}, {"\\chi", "χ"}, {"\\psi", "ψ"}, {"\\omega", "ω"},
  // Variant forms
  {"\\varepsilon", "ε"}, {"\\vartheta", "ϑ"}, {"\\varpi", "ϖ"},
  {"\\varrho", "ϱ"}, {"\\varsigma", "ς"}, {"\\varphi", "ϕ"},
  // Uppercase Greek letters
  {"\\Alpha", "Α"}, {"\\Beta", "Β"}, {"\\Gamma", "Γ"}, {"\\Delta", "Δ"},
  {"\\Epsilon", "Ε"}, {"\\Zeta", "Ζ"}, {"\\Eta", "Η"}, {"\\Theta", "Θ"},
  {"\\Iota", "Ι"}, {"\\Kappa", "Κ"}, {"\\Lambda", "Λ"}, {"\\Mu", "Μ"},
  {"\\Nu", "Ν"}, {"\\Xi", "Ξ"}, {"\\Omicron", "Ο"}, {"\\Pi", "Π"},
  {"\\Rho", "Ρ"}, {"\\Sigma", "Σ"}, {"\\Tau", "Τ"}, {"\\Upsilon", "Υ"},
  {"\\Phi", "Φ"}, {"\\Chi", "Χ"}, {"\\Psi", "Ψ"}, {"\\Omega", "Ω"}
};

/// `s` with each character replaced by its Unicode subscript, where one exists.
inline std::string toSubscript(const std::string& s) {
  std::string sub;
  for (const char c : s) {
    std::string key(1, c);
    auto it = subscriptLookup.find(key);
    if (it != subscriptLookup.end()) {
      sub += it->second;
    } else {
      sub += c;  // Keep original character if no subscript exists
    }
  }
  return sub;
}

/// `s` with each LaTeX Greek-letter command replaced by the corresponding Unicode letter.
/// Unrecognized backslash commands are left as they are.
inline std::string greekToUtf8(const std::string& s) {
  std::string result;
  size_t pos = 0;

  while (pos < s.length()) {
    // Check if we're at a backslash (potential LaTeX command)
    if (s[pos] == '\\') {
      // Find the end of the command (next non-letter character)
      size_t end = pos + 1;
      while (end < s.length() && std::isalpha(s[end])) {
        end++;
      }

      std::string command = s.substr(pos, end - pos);
      auto it = greekLookup.find(command);

      if (it != greekLookup.end()) {
        result += it->second;
        pos = end;
      } else {
        // Not a recognized Greek letter, keep the backslash
        result += s[pos];
        pos++;
      }
    } else {
      result += s[pos];
      pos++;
    }
  }

  return result;
}

/// `s` with each character replaced by its Unicode superscript, where one exists.
inline std::string toSuperscript(const std::string& s) {
  std::string sup;
  for (const char c : s) {
    std::string key(1, c);
    auto it = superscriptLookup.find(key);
    if (it != superscriptLookup.end()) {
      sup += it->second;
    } else {
      sup += c;  // Keep original character if no superscript exists
    }
  }
  return sup;
}

/// Render simple LaTeX as Unicode for terminal output: Greek letters, then `_x` / `_{...}`
/// subscripts and `^x` / `^{...}` superscripts. The result is wrapped in the ANSI bold
/// on/off sequences. Returns `s` unchanged unless both the environment and the C locale
/// claim UTF-8.
inline std::string latexToUtf8(const std::string& s) {
  if (!envClaimsUTF8() || !localClaimsUTF8()) return s;

  // First, convert Greek letters
  std::string result = greekToUtf8(s);

  // Process subscripts and superscripts
  std::string processed;
  size_t pos = 0;

  while (pos < result.length()) {
    // Check for subscript: _{...} or _x
    if (result[pos] == '_' && pos + 1 < result.length()) {
      pos++;  // Skip the underscore

      if (result[pos] == '{') {
        // Braced subscript: _{content}
        pos++;  // Skip opening brace
        size_t start = pos;
        size_t depth = 1;

        // Find matching closing brace
        while (pos < result.length() && depth > 0) {
          if (result[pos] == '{') depth++;
          else if (result[pos] == '}') depth--;
          if (depth > 0) pos++;
        }

        std::string content = result.substr(start, pos - start);
        processed += toSubscript(content);
        pos++;  // Skip closing brace
      } else {
        // Single character subscript: _x
        std::string key(1, result[pos]);
        auto it = subscriptLookup.find(key);
        if (it != subscriptLookup.end()) {
          processed += it->second;
        } else {
          processed += result[pos];
        }
        pos++;
      }
    }
    // Check for superscript: ^{...} or ^x
    else if (result[pos] == '^' && pos + 1 < result.length()) {
      pos++;  // Skip the caret

      if (result[pos] == '{') {
        // Braced superscript: ^{content}
        pos++;  // Skip opening brace
        size_t start = pos;
        size_t depth = 1;

        // Find matching closing brace
        while (pos < result.length() && depth > 0) {
          if (result[pos] == '{') depth++;
          else if (result[pos] == '}') depth--;
          if (depth > 0) pos++;
        }

        std::string content = result.substr(start, pos - start);
        processed += toSuperscript(content);
        pos++;  // Skip closing brace
      } else {
        // Single character superscript: ^x
        std::string key(1, result[pos]);
        auto it = superscriptLookup.find(key);
        if (it != superscriptLookup.end()) {
          processed += it->second;
        } else {
          processed += result[pos];
        }
        pos++;
      }
    }
    else {
      processed += result[pos];
      pos++;
    }
  }

  return "\x1b[1m" + processed + "\x1b[22m";
}

/// Debugging aid that tracks which containers hold each named element and logs every
/// insertion and removal. The containers themselves stay owned by the caller; this class
/// stores only names, so an element's entry can list the same container more than once if
/// it was inserted more than once.
template<typename KeyType, typename ElementType, typename ElementHash, typename ElementEq>
class OwnershipManager {
  public:

    /// Element name -> the names of the containers currently holding it.
    std::unordered_map<std::string, std::vector<std::string>> references{};

    /// Insert `element` under `key` into `container` and record the ownership.
    auto insert(
      std::string elementName,
      std::string containerName,
      std::unordered_map<KeyType, ElementType, ElementHash, ElementEq> &container,
      KeyType &key,
      ElementType &element
    ) {
      CLOG(INFO_LEVEL, "Inserting an element ", elementName, " into container ", containerName);
      if (references.find(elementName) == references.end()) {
        references.insert_or_assign(elementName, std::vector<std::string>{containerName});
      } else {
        references[elementName].push_back(containerName);
      }
      return container.insert(key, element);
    }

    /// Insert `element` into a set-shaped `container` and record the ownership.
    auto insert(
      std::string elementName,
      std::string containerName,
      std::unordered_set<ElementType, ElementHash, ElementEq> &container,
      const ElementType &element) {
      CLOG(INFO_LEVEL, "Inserting an element ", elementName, " into container ", containerName);
      if (references.find(elementName) == references.end()) {
        references.insert_or_assign(elementName, std::vector<std::string>{containerName});
      } else {
        references[elementName].push_back(containerName);
      }
      return container.insert(element);
    }

    /// Erase `key` from `container` and drop one recorded ownership entry.
    auto erase(
      std::string elementName,
      std::string containerName,
      std::unordered_map<KeyType, ElementType, ElementHash, ElementEq> &container,
      KeyType &key
    ) {
      CLOG(INFO_LEVEL, "Removing an element ", elementName, " from container ", containerName);
      auto it = std::find(references[elementName].begin(), references[elementName].end(), containerName);
      if (it != references[elementName].end()) {
        references[elementName].erase(it);
      }
      return container.erase(key);
    }

    /// Erase `element` from a set-shaped `container` and drop one recorded ownership entry.
    auto erase(
      std::string elementName,
      std::string containerName,
      std::unordered_set<ElementType, ElementHash, ElementEq> &container,
      const ElementType &element
      ) {
      CLOG(INFO_LEVEL, "Removing an element ", elementName, " from container ", containerName);
      auto it = std::find(references[elementName].begin(), references[elementName].end(), containerName);
      if (it != references[elementName].end()) {
        references[elementName].erase(it);
      }
      return container.erase(element);
    }

    /// Log every element that is still held, with its container names sorted.
    void showReferences() {
      for (auto &[elementName, containerNames] : references) {
        if (containerNames.empty()) continue;
        CLOG(INFO_LEVEL, elementName, ":");
        std::sort(containerNames.begin(), containerNames.end(), [](const std::string &a, const std::string &b) {
          return a < b;
        });
        for (auto &containerName : containerNames) {
          CLOG(INFO_LEVEL, "      - ", containerName);
        }
      }
    }
};

#endif //TESSERA_UTILS_H