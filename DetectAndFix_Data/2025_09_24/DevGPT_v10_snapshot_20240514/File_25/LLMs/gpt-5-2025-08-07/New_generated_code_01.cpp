#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <filesystem>

// Security-hardening constants
static constexpr std::uintmax_t MAX_INPUT_SIZE = 10 * 1024 * 1024; // 10 MB

// Safe character classification wrappers to avoid UB with signed char
inline bool isSpace(unsigned char c) { return std::isspace(c) != 0; }
inline bool isAlphaNum(unsigned char c) { return std::isalnum(c) != 0; }
inline bool isQuote(unsigned char c) { return c == '"' || c == '\''; }

// Trim helpers
static inline void ltrim_inplace(std::string& s) {
    size_t i = 0;
    while (i < s.size() && isSpace(static_cast<unsigned char>(s[i]))) ++i;
    if (i > 0) s.erase(0, i);
}
static inline void rtrim_inplace(std::string& s) {
    size_t n = s.size();
    while (n > 0 && isSpace(static_cast<unsigned char>(s[n-1]))) --n;
    s.resize(n);
}
static inline void trim_inplace(std::string& s) {
    rtrim_inplace(s);
    ltrim_inplace(s);
}

// Neutralize control chars to mitigate log/console injection (CWE-117)
std::string neutralizeControlChars(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c < 0x20 || c == 0x7F) {
            // Keep common whitespace, but neutralize CR and others
            if (c == '\n' || c == '\t') {
                out.push_back(c);
            } else {
                static const char hex[] = "0123456789ABCDEF";
                out += "\\x";
                out.push_back(hex[(c >> 4) & 0xF]);
                out.push_back(hex[c & 0xF]);
            }
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

// Safe file reading with size limit and robust error handling (CWE-400/770, CWE-20)
std::string readFileIntoRawString(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path p(path);

    std::error_code ec;
    if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) {
        throw std::runtime_error("Input file does not exist or is not a regular file: " + path);
    }
    std::uintmax_t sz = fs::file_size(p, ec);
    if (ec) throw std::runtime_error("Failed to query file size: " + path + " (" + ec.message() + ")");
    if (sz > MAX_INPUT_SIZE) {
        throw std::runtime_error("Input file too large (" + std::to_string(sz) + " bytes), cap is " + std::to_string(MAX_INPUT_SIZE));
    }

    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open input file: " + path);

    std::string data;
    data.resize(static_cast<size_t>(sz));
    if (sz > 0 && !ifs.read(&data[0], static_cast<std::streamsize>(sz))) {
        throw std::runtime_error("Failed to read input file: " + path);
    }
    return data;
}

// Safe file write with basic atomicity (write temp then rename)
void writeToTxtFile(const std::string& content, const std::string& path) {
    namespace fs = std::filesystem;
    fs::path target(path);
    fs::path tmp = target;
    tmp += ".tmp";

    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) throw std::runtime_error("Failed to open temp output file: " + tmp.string());
        ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!ofs) throw std::runtime_error("Failed to write to temp output file: " + tmp.string());
        ofs.flush();
        if (!ofs) throw std::runtime_error("Failed to flush temp output file: " + tmp.string());
    }

    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        // On Windows, rename fails if target exists. Remove then rename.
        fs::remove(target, ec); // ignore error
        fs::rename(tmp, target, ec);
        if (ec) {
            // Clean up temp on failure
            fs::remove(tmp, ec);
            throw std::runtime_error("Failed to finalize output file: " + target.string());
        }
    }
}

// Function to remove leading whitespace at the beginning of the entire text and also normalize leading whitespace per line
void removeLeadingWhitespace(std::string& text) {
    // Trim start of entire text
    ltrim_inplace(text);

    // Remove leading spaces for each line
    std::string out;
    out.reserve(text.size());
    bool atLineStart = true;
    for (unsigned char c : text) {
        if (atLineStart && isSpace(c) && c != '\n' && c != '\r' && c != '\t') {
            // skip leading spaces
            continue;
        }
        out.push_back(static_cast<char>(c));
        if (c == '\n' || c == '\r') atLineStart = true; else atLineStart = false;
    }
    text.swap(out);
}

// Remove line if, after optional leading whitespace, it starts with `sequence`
void removeLineStartingWith(std::string& text, const std::string& sequence) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0, n = text.size();
    while (i < n) {
        size_t lineStart = i;
        // find line end
        size_t lineEnd = i;
        while (lineEnd < n && text[lineEnd] != '\n') ++lineEnd;

        // inspect line
        size_t j = lineStart;
        while (j < lineEnd && isSpace(static_cast<unsigned char>(text[j])) && text[j] != '\n' && text[j] != '\r') ++j;

        bool remove = false;
        if (j + sequence.size() <= lineEnd) {
            if (text.compare(j, sequence.size(), sequence) == 0) {
                // If followed by whitespace, '=' or end-of-line, consider it a match
                size_t k = j + sequence.size();
                if (k == lineEnd || text[k] == '=' || isSpace(static_cast<unsigned char>(text[k]))) {
                    remove = true;
                }
            }
        }

        if (!remove) {
            out.append(text, lineStart, (lineEnd - lineStart));
            if (lineEnd < n) out.push_back('\n');
        } else {
            // skip line (effectively remove)
            if (lineEnd < n) {
                // keep newline to maintain line count if desired; here we drop it
            }
        }

        i = (lineEnd < n) ? (lineEnd + 1) : lineEnd;
    }
    text.swap(out);
}

std::string removeQuotes(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char c : input) {
        if (c != '"') out.push_back(static_cast<char>(c));
    }
    return out;
}

void ensureNewlineAfterBrace(std::string& text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        out.push_back(c);
        if ((c == '{' || c == '}')) {
            // If next char exists and isn't a newline, insert newline
            if (i + 1 < text.size()) {
                char next = text[i + 1];
                if (next != '\n') out.push_back('\n');
            } else {
                out.push_back('\n');
            }
        }
    }
    text.swap(out);
}

void insertNewlineBetweenBrackets(std::string& input) {
    // Not used in main, but implemented safely
    std::string out;
    out.reserve(input.size() + 16);
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        out.push_back(c);
        if ((c == '}' || c == ']') && i + 1 < input.size()) {
            char next = input[i + 1];
            if (next == '{' || next == '[') out.push_back('\n');
        }
    }
    input.swap(out);
}

void removeNewlineAfterEquals(std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        out.push_back(c);
        if (c == '=') {
            // Skip immediate CR/LF after '='
            size_t j = i + 1;
            if (j < text.size() && (text[j] == '\r' || text[j] == '\n')) {
                // Skip CR
                if (text[j] == '\r') ++j;
                // Skip single LF
                if (j < text.size() && text[j] == '\n') ++j;
                i = j - 1;
            }
        }
    }
    text.swap(out);
}

// Extract top-level members delimited by balanced braces { ... } while respecting quotes.
// Input could be like: {...}\n{...}\n or nested inside other wrappers.
std::vector<std::string> extractIndividualMembers(const std::string& input) {
    std::vector<std::string> members;
    int depth = 0;
    bool inString = false;
    char stringDelim = '\0';
    bool escape = false;

    size_t start = std::string::npos;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (inString) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == stringDelim) {
                inString = false;
            }
            continue;
        } else {
            if (c == '"' || c == '\'') {
                inString = true;
                stringDelim = c;
                continue;
            }
            if (c == '{') {
                if (depth == 0) start = i;
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth < 0) {
                    // Unbalanced closing brace; ignore
                    depth = 0;
                } else if (depth == 0 && start != std::string::npos) {
                    members.emplace_back(input.substr(start, i - start + 1));
                    start = std::string::npos;
                }
            }
        }
    }
    return members;
}

// Class to represent a Factorio recipe
class Recipe {
public:
    explicit Recipe(const std::string& member)
        : rawMember(removeCraftingMachineTint(member)) {}

    std::string getName() const;
    std::string getIngredients() const;
    std::string toString() const;

private:
    std::string extractNestedBraces(const std::string& input) const;
    std::string parseIngredients(const std::string& ingredientsData) const;
    std::string removeCraftingMachineTint(const std::string& inputMember) const;

    std::string rawMember;
};

// Extract the first balanced brace block after a keyword like "ingredients"
std::string Recipe::extractNestedBraces(const std::string& input) const {
    int depth = 0;
    bool inString = false;
    char stringDelim = '\0';
    bool escape = false;
    size_t start = std::string::npos;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (inString) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == stringDelim) inString = false;
            continue;
        } else {
            if (c == '"' || c == '\'') {
                inString = true;
                stringDelim = c;
                continue;
            }
            if (c == '{') {
                if (depth == 0) start = i;
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth < 0) {
                    depth = 0;
                } else if (depth == 0 && start != std::string::npos) {
                    return input.substr(start, i - start + 1);
                }
            }
        }
    }
    return std::string();
}

// Remove any "crafting_machine_tint = {...}" blocks safely
std::string Recipe::removeCraftingMachineTint(const std::string& inputMember) const {
    std::string out;
    out.reserve(inputMember.size());

    const std::string key = "crafting_machine_tint";
    size_t i = 0, n = inputMember.size();
    while (i < n) {
        // Skip whitespace
        size_t j = i;
        while (j < n && isSpace(static_cast<unsigned char>(inputMember[j]))) ++j;

        bool matched = false;
        if (j + key.size() < n && inputMember.compare(j, key.size(), key) == 0) {
            size_t k = j + key.size();
            while (k < n && isSpace(static_cast<unsigned char>(inputMember[k]))) ++k;
            if (k < n && inputMember[k] == '=') {
                ++k;
                while (k < n && isSpace(static_cast<unsigned char>(inputMember[k]))) ++k;
                if (k < n && inputMember[k] == '{') {
                    // Skip balanced braces
                    int depth = 0;
                    bool inStr = false;
                    char delim = '\0';
                    bool esc = false;
                    size_t startSkip = i; // from original whitespace start
                    size_t m = k;
                    for (; m < n; ++m) {
                        char c = inputMember[m];
                        if (inStr) {
                            if (esc) esc = false;
                            else if (c == '\\') esc = true;
                            else if (c == delim) inStr = false;
                        } else {
                            if (c == '"' || c == '\'') { inStr = true; delim = c; }
                            else if (c == '{') ++depth;
                            else if (c == '}') {
                                --depth;
                                if (depth == 0) { ++m; break; }
                            }
                        }
                    }
                    // Skip until end of line (optional)
                    while (m < n && inputMember[m] != '\n') ++m;
                    if (m < n && inputMember[m] == '\n') ++m;
                    i = m;
                    matched = true;
                }
            }
        }

        if (!matched) {
            out.push_back(inputMember[i]);
            ++i;
        }
    }

    return out;
}

std::string Recipe::parseIngredients(const std::string& ingredientsData) const {
    // For security: return a normalized/minimally processed view without executing regex.
    // Just trim and return content inside the outermost braces if present.
    std::string s = ingredientsData;
    trim_inplace(s);
    if (!s.empty() && s.front() == '{' && s.back() == '}') {
        // Optionally, we could further sanitize inner control chars (keep it simple)
        return s;
    }
    return s;
}

std::string Recipe::getName() const {
    // Find name = "..."
    const std::string key = "name";
    size_t pos = 0;
    while (pos < rawMember.size()) {
        // find 'n' of "name"
        size_t k = rawMember.find(key, pos);
        if (k == std::string::npos) break;
        size_t i = k + key.size();
        // optionally require '='
        while (i < rawMember.size() && isSpace(static_cast<unsigned char>(rawMember[i]))) ++i;
        if (i < rawMember.size() && rawMember[i] == '=') {
            ++i;
            while (i < rawMember.size() && isSpace(static_cast<unsigned char>(rawMember[i]))) ++i;
            if (i < rawMember.size() && (rawMember[i] == '"' || rawMember[i] == '\'')) {
                char delim = rawMember[i++];
                std::string name;
                bool esc = false;
                for (; i < rawMember.size(); ++i) {
                    char c = rawMember[i];
                    if (esc) { name.push_back(c); esc = false; }
                    else if (c == '\\') esc = true;
                    else if (c == delim) break;
                    else name.push_back(c);
                }
                trim_inplace(name);
                return name;
            } else {
                // unquoted name; read until whitespace or comma
                std::string name;
                while (i < rawMember.size() && !isSpace(static_cast<unsigned char>(rawMember[i])) && rawMember[i] != ',' && rawMember[i] != '\n' && rawMember[i] != '}') {
                    name.push_back(rawMember[i++]);
                }
                trim_inplace(name);
                if (!name.empty()) return name;
            }
        }
        pos = k + 1;
    }
    return std::string("UNKNOWN_NAME");
}

std::string Recipe::getIngredients() const {
    // Find "ingredients" then extract a balanced brace block after '='
    const std::string key = "ingredients";
    size_t pos = rawMember.find(key);
    if (pos == std::string::npos) return std::string("{}");
    size_t i = pos + key.size();
    while (i < rawMember.size() && isSpace(static_cast<unsigned char>(rawMember[i]))) ++i;
    if (i >= rawMember.size() || rawMember[i] != '=') return std::string("{}");
    ++i;
    while (i < rawMember.size() && isSpace(static_cast<unsigned char>(rawMember[i]))) ++i;

    std::string tail = rawMember.substr(i);
    std::string nested = extractNestedBraces(tail);
    if (nested.empty()) return std::string("{}");
    return parseIngredients(nested);
}

std::string Recipe::toString() const {
    std::ostringstream oss;
    oss << "name: " << getName() << "\ningredients: " << getIngredients();
    return oss.str();
}

// Function to generate output from individual members
std::string generateOutput(const std::vector<Recipe>& recipes) {
    std::ostringstream rawStringStream;
    rawStringStream << "{\n";
    for (size_t i = 0; i < recipes.size(); ++i) {
        // Neutralize to prevent log/console injection (CWE-117)
        std::string nameSafe = neutralizeControlChars(removeQuotes(recipes[i].getName()));
        std::string ingSafe  = neutralizeControlChars(removeQuotes(recipes[i].getIngredients()));
        rawStringStream << nameSafe << '\n' << ingSafe << "\n\n";
    }
    rawStringStream << "}\n";
    return rawStringStream.str();
}

int main() {
    try {
        const std::string filename = "TextFile1.txt";
        std::string rawString = readFileIntoRawString(filename);

        // Cleaning up raw string
        removeLeadingWhitespace(rawString);
        ensureNewlineAfterBrace(rawString);
        removeLeadingWhitespace(rawString);

        std::vector<std::string> catsToRemove = {
            "type", "category", "enabled", "order", "allow_decomposition",
            "main_product", "subgroup", "requester_paste_multiplier", "icon"
        };
        for (const auto& category : catsToRemove) {
            removeLineStartingWith(rawString, category);
        }
        removeNewlineAfterEquals(rawString);

        // Extract individual members (recipes)
        std::vector<std::string> individualMembers = extractIndividualMembers(rawString);

        // Create Recipe objects
        std::vector<Recipe> recipes;
        recipes.reserve(individualMembers.size());
        for (const auto& member : individualMembers) {
            recipes.emplace_back(member);
        }

        // Generate output
        std::string output = generateOutput(recipes);

        // Output to console (neutralized)
        std::cout << output;

        // Output to file (neutralized already above)
        writeToTxtFile(output, "roughCut.txt");

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << neutralizeControlChars(ex.what()) << "\n";
        return 1;
    }
}