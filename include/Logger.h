// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <cctype>

#ifdef TESSERA_HAS_OSYNCSTREAM
  #include <syncstream>   // C++20
#endif

#if defined(_WIN32)
  #include <io.h>
  #define isatty _isatty
  #define fileno _fileno
#else
  #include <unistd.h>
#endif

const short int DEBUG_LEVEL = 10;
const short int INFO_LEVEL = 20;
const short int WARN_LEVEL = 30;
const short int ERROR_LEVEL = 40;
const short int CRITICAL_LEVEL = 50;

/// Severity-filtered logger. Records go to stderr as
/// `<time> [<LEVEL>] <file>:<line> <func> :: <message>`, colorized when stderr is an
/// interactive terminal. Every member is static; there is no instance state. Use the CLOG
/// macro rather than calling log() directly, so that __FILE__/__func__/__LINE__ are the
/// caller's.
class Logger {

 public:
  /// The active threshold, once resolved. Set it to override the LOG_LEVEL environment
  /// variable; leave it empty to have getLevel() resolve it on first use.
  static std::optional<short int> LEVEL;
  /// The current local time as "YYYY-MM-DD HH:MM:SS".
  static std::string getTime();
  /// The active threshold, resolved on first call from LEVEL, else from the LOG_LEVEL
  /// environment variable, else INFO_LEVEL. The result is cached in LEVEL.
  static short int getLevel();
  /// The name of a severity level ("DEBUG" ... "CRITICAL"), or "UNKNOWN".
  static std::string nameLevel(short int level);

  /// Write one already-formatted record to stderr.
  static void emit(short int level,
                   const std::string &filename,
                   std::string func,
                   const int lineno,
                   const std::string &message);

  /// `absolute` with everything up to and including `root` stripped; `absolute` unchanged
  /// if `root` does not occur in it.
  static std::string makeRelative(const std::string &absolute, const std::string &root);

    /// Stream `args` into one message and emit it, if `level` passes getLevel().
    /// The filename is reported relative to SOURCES_ROOT.
    template<typename... Args>
    static void log(const short int level,
                const std::string &filename,
                std::string func,
                int lineno,
                Args&&... args) {
      if (level < Logger::getLevel()) return;

      std::ostringstream message;
      (message << ... << args);

      std::string as_str = message.str();
      std::string relative = Logger::makeRelative(filename, SOURCES_ROOT);
      emit(level, relative, std::move(func), lineno, as_str);
    }

 private:
  static bool envTruthy(const char* v) {
    if (!v) return false;
    while (*v && std::isspace(static_cast<unsigned char>(*v))) ++v;
    if (!*v) return false;
    // "0", "false", "no" and "off" are false; anything else is true
    if (v[0] == '0' && v[1] == '\0') return false;
    auto lower = [](unsigned char c){ return static_cast<char>(std::tolower(c)); };
    std::string s;
    for (const char* p = v; *p; ++p) s.push_back(lower(static_cast<unsigned char>(*p)));
    return !(s == "false" || s == "no" || s == "off");
  }

  static bool envFalsy(const char* v) {
    if (!v) return false;
    while (*v && std::isspace(static_cast<unsigned char>(*v))) ++v;
    if (!*v) return false;
    if (v[0] == '0' && v[1] == '\0') return true;
    auto lower = [](unsigned char c){ return static_cast<char>(std::tolower(c)); };
    std::string s;
    for (const char* p = v; *p; ++p) s.push_back(lower(static_cast<unsigned char>(*p)));
    return (s == "false" || s == "no" || s == "off");
  }

  static bool shouldColorize() {
    // Explicit overrides win.
    if (envTruthy(std::getenv("TESSERA_LOG_COLOR"))) return true;
    if (envFalsy(std::getenv("TESSERA_LOG_COLOR")))  return false;

    // The conventional opt-out.
    if (std::getenv("NO_COLOR")) return false;

    // If stderr isn't a terminal, don't inject escape codes.
    if (!isatty(fileno(stderr))) return false;

    const char* term = std::getenv("TERM");
    if (!term || std::string_view(term) == "dumb") return false;

    return true;
  }

  static std::string_view levelStyle(short int level) {
    // ANSI SGR codes. dim: 2, bold: 1; colors: 31 red, 32 green, 33 yellow, 36 cyan,
    // 90 bright black.
    if (level >= CRITICAL_LEVEL) return "\x1b[1;31m"; // bold red
    if (level >= ERROR_LEVEL)    return "\x1b[31m  ";   // red
    if (level >= WARN_LEVEL)     return "\x1b[33m    ";   // yellow
    if (level >= INFO_LEVEL)     return "\x1b[32m    ";   // green
    return "\x1b[2;90m   ";                               // dim gray for debug/trace-ish
  }

  static constexpr std::string_view resetStyle() { return "\x1b[0m"; }
};

// --- emit(), with colorization ---
inline void Logger::emit(short int level,
                         const std::string &filename,
                         std::string func,
                         const int lineno,
                         const std::string &message) {
  const bool color = shouldColorize();

  const std::string ts = getTime();
  const std::string lvl = nameLevel(level);

#if TESSERA_HAS_OSYNCSTREAM
  std::osyncstream out(std::cerr);
  if (color) out << levelStyle(level);
  out << ts << " [" << lvl << "] ";
  if (color) out << resetStyle();
  out << filename << ":" << lineno << " " << func << " :: "
      << message;
  out << '\n';
#else
  // Fallback: not atomic across threads, but works everywhere.
  if (color) std::cerr << levelStyle(level);
  std::cerr << ts << " [" << lvl << "] ";
  if (color) std::cerr << resetStyle();
  std::cerr << filename << ":" << lineno << " " << func << " :: "
            << message;
  std::cerr << '\n';
#endif

}

#ifdef TESSERA_VERBOSE
  #define CLOG(level, ...) Logger::log(level, __FILE__, __func__, __LINE__, __VA_ARGS__)
#else
  #define CLOG(level, ...) if ((level) == CRITICAL_LEVEL || (level) == ERROR_LEVEL) { Logger::log(level, __FILE__, __func__, __LINE__, __VA_ARGS__); }
#endif

#endif // LOGGER_HPP
