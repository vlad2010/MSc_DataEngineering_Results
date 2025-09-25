#include <pcre.h>
#include <climits>
#include <cstddef>
#include <iostream>

namespace {
    // Tune these based on your environment and acceptable worst-case time.
    constexpr unsigned DEFAULT_MATCH_LIMIT = 100000;           // Limits total backtracking steps
    constexpr unsigned DEFAULT_RECURSION_LIMIT = 10000;        // Limits recursion depth (stack usage)
    constexpr int OVECSIZE = 30;                               // Must be a multiple of 3 (captures + full match)
    static_assert(OVECSIZE % 3 == 0, "ovector size must be a multiple of 3");
}

// Secure wrapper around pcre_exec: caller must supply subject length to avoid strlen on untrusted data.
bool matchRegex(const pcre* re, const char* subject, size_t subjectLen) {
    // Basic validation (CWE-476)
    if (re == nullptr || subject == nullptr) {
        std::cerr << "matchRegex: null pointer argument." << std::endl;
        return false;
    }

    // Prevent integer truncation (CWE-190/CWE-197)
    if (subjectLen > static_cast<size_t>(INT_MAX)) {
        std::cerr << "matchRegex: subject too long." << std::endl;
        return false;
    }
    const int ilen = static_cast<int>(subjectLen);

    // Set match/backtracking limits to mitigate ReDoS (CWE-1333/CWE-400) and stack overuse
    pcre_extra extra{};
    extra.flags = 0;
#ifdef PCRE_EXTRA_MATCH_LIMIT
    extra.flags |= PCRE_EXTRA_MATCH_LIMIT;
    extra.match_limit = DEFAULT_MATCH_LIMIT;
#endif
#ifdef PCRE_EXTRA_MATCH_LIMIT_RECURSION
    extra.flags |= PCRE_EXTRA_MATCH_LIMIT_RECURSION;
    extra.match_limit_recursion = DEFAULT_RECURSION_LIMIT;
#endif

    int ovector[OVECSIZE];

    // Note: rc == 0 means the ovector was too small to hold all captures, but a match occurred.
    const int rc = pcre_exec(
        re,
        (extra.flags != 0 ? &extra : nullptr),
        subject,
        ilen,
        /*startoffset*/ 0,
        /*options*/ 0,
        ovector,
        OVECSIZE
    );

    if (rc >= 0) {
        // Match found (even if rc == 0 due to limited ovector space)
        // Avoid noisy I/O in library code; return result instead.
        return true;
    }

    if (rc == PCRE_ERROR_NOMATCH) {
        return false;
    }

    // Handle limit-related errors distinctly to aid tuning/alerts
    if (rc == PCRE_ERROR_MATCHLIMIT || rc == PCRE_ERROR_RECURSIONLIMIT) {
        std::cerr << "matchRegex: aborted due to regex limits (rc=" << rc << ")." << std::endl;
        return false;
    }

    // Other errors (compilation/runtime/environment); in production, consider mapping to generic error
    std::cerr << "matchRegex: regex engine error (rc=" << rc << ")." << std::endl;
    return false;
}

/*
Usage notes:
- Ensure 'subject' is a valid buffer and provide its exact length in subjectLen.
- Consider compiling the pattern with options that reduce backtracking risk where possible,
  or using pcre_dfa_exec when only a boolean match is needed (DFA avoids catastrophic backtracking).
- Consider migrating to PCRE2 for improved API (64-bit lengths, contexts, better limits, JIT control).
*/