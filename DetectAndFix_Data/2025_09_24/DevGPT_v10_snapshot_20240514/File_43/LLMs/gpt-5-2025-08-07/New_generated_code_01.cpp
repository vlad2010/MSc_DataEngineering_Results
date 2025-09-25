#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <regex>
#include <filesystem>
#include <limits>
#include <cmath>
#include <unordered_set>

class SVGWriter {
public:
    SVGWriter(const std::string &filename, int width, int height)
    {
        // Validate width/height
        if (width < 0 || height < 0) {
            throw std::invalid_argument("SVG dimensions must be non-negative");
        }

        // Validate filename: only simple names, no path separators, enforce .svg
        const std::regex safeName(R"(^[A-Za-z0-9._-]{1,128}\.svg$)");
        if (!std::regex_match(filename, safeName)) {
            throw std::invalid_argument("Unsafe filename; only [A-Za-z0-9._-] with .svg extension allowed");
        }

        std::filesystem::path p(filename);

        // Best-effort symlink check (TOCTTOU note: not atomic)
        if (std::filesystem::exists(p) && std::filesystem::is_symlink(p)) {
            throw std::runtime_error("Refusing to open symlinked path for output");
        }

        // Open file; set exceptions to catch write failures
        svgFile.open(p, std::ios::out | std::ios::trunc);
        svgFile.exceptions(std::ofstream::failbit | std::ofstream::badbit);

        // Write SVG header (attributes are integers; safe)
        svgFile << "<svg width=\"" << width << "\" height=\"" << height
                << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
        headerWritten = true;
    }

    ~SVGWriter() noexcept {
        try {
            if (svgFile.is_open()) {
                if (headerWritten) {
                    svgFile << "</svg>\n";
                }
                svgFile.close();
            }
        } catch (...) {
            // Do not throw from destructor
        }
    }

    void writeRect(int width, int height, const std::string &fillColor) {
        if (width < 0 || height < 0) {
            throw std::invalid_argument("Rectangle dimensions must be non-negative");
        }
        const std::string fill = sanitizeColor(fillColor);
        svgFile << "<rect width=\"" << width
                << "\" height=\"" << height
                << "\" fill=\"" << escapeXmlAttr(fill) << "\"/>\n";
    }

    void writeCircle(double cx, double cy, double r,
                     const std::string &strokeColor,
                     const std::string &fillColor,
                     double strokeWidth) {
        if (!isFinite(cx) || !isFinite(cy) || !isFinite(r) || !isFinite(strokeWidth)) {
            throw std::invalid_argument("Circle parameters must be finite numbers");
        }
        if (r < 0.0 || strokeWidth < 0.0) {
            throw std::invalid_argument("Radius and strokeWidth must be non-negative");
        }
        const std::string stroke = sanitizeColor(strokeColor);
        const std::string fill   = sanitizeColor(fillColor);

        svgFile << "<circle cx=\"" << cx
                << "\" cy=\"" << cy
                << "\" r=\"" << r
                << "\" stroke=\"" << escapeXmlAttr(stroke)
                << "\" fill=\"" << escapeXmlAttr(fill)
                << "\" stroke-width=\"" << strokeWidth << "\" />\n";
    }

    void writeMarker(const std::string &markerId) {
        const std::string id = sanitizeId(markerId);
        svgFile << "<defs>\n";
        svgFile << "  <marker style=\"overflow:visible\" id=\"" << escapeXmlAttr(id)
                << "\" refX=\"0.0\" refY=\"0.0\" orient=\"auto\">\n";
        svgFile << "    <path transform=\"scale(0.2) translate(6,0)\" "
                   "style=\"fill-rule:evenodd;fill:context-stroke;stroke:context-stroke;stroke-width:1.0pt\" "
                   "d=\"M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z \" />\n";
        svgFile << "  </marker>\n";
        svgFile << "</defs>\n";
    }

    void writeLine(double x1, double y1, double x2, double y2,
                   const std::string &strokeColor, double strokeWidth,
                   const std::string &markerStart) {
        if (!isFinite(x1) || !isFinite(y1) || !isFinite(x2) || !isFinite(y2) ||
            !isFinite(strokeWidth)) {
            throw std::invalid_argument("Line parameters must be finite numbers");
        }
        if (strokeWidth < 0.0) {
            throw std::invalid_argument("strokeWidth must be non-negative");
        }

        const std::string stroke = sanitizeColor(strokeColor);
        const std::string id     = sanitizeId(markerStart);

        // Construct marker-start URL safely with sanitized ID
        std::string markerUrl = "url(#" + id + ")";

        svgFile << "<line x1=\"" << x1
                << "\" y1=\"" << y1
                << "\" x2=\"" << x2
                << "\" y2=\"" << y2
                << "\" stroke=\"" << escapeXmlAttr(stroke)
                << "\" stroke-width=\"" << strokeWidth
                << "\" marker-start=\"" << escapeXmlAttr(markerUrl) << "\" />\n";
    }

private:
    std::ofstream svgFile;
    bool headerWritten{false};

    static bool isFinite(double v) {
        return std::isfinite(v);
    }

    // Escape XML attribute special characters
    static std::string escapeXmlAttr(const std::string &in) {
        std::string out;
        out.reserve(in.size());
        for (char c : in) {
            switch (c) {
                case '&':  out += "&amp;";  break;
                case '<':  out += "&lt;";   break;
                case '>':  out += "&gt;";   break;
                case '"':  out += "&quot;"; break;
                case '\'': out += "&apos;"; break;
                default:   out += c;        break;
            }
        }
        return out;
    }

    // Allow only safe CSS colors: hex (#RGB, #RRGGBB, #RRGGBBAA) or a whitelist of named colors
    static std::string sanitizeColor(const std::string &color) {
        static const std::regex hexShort(R"(^#([0-9a-fA-F]{3})$)");
        static const std::regex hexLong(R"(^#([0-9a-fA-F]{6})$)");
        static const std::regex hexAlpha(R"(^#([0-9a-fA-F]{8})$)");

        if (std::regex_match(color, hexShort) ||
            std::regex_match(color, hexLong)  ||
            std::regex_match(color, hexAlpha)) {
            return color;
        }

        // Minimal whitelist of named colors (extend as needed)
        static const std::unordered_set<std::string> named = {
            "black","silver","gray","white","maroon","red","purple","fuchsia",
            "green","lime","olive","yellow","navy","blue","teal","aqua",
            "orange"
        };
        if (named.count(lowercopy(color)) > 0) {
            return lowercopy(color);
        }

        // Fallback to a safe default
        return std::string("#000000");
    }

    static std::string lowercopy(const std::string &s) {
        std::string t; t.reserve(s.size());
        for (unsigned char c : s) t.push_back(static_cast<char>(std::tolower(c)));
        return t;
    }

    // Constrain ID to a safe XML Name-like pattern
    static std::string sanitizeId(const std::string &id) {
        static const std::regex idPattern(R"(^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$)");
        if (!std::regex_match(id, idPattern)) {
            throw std::invalid_argument("Invalid ID: must match XML Name pattern");
        }
        return id;
    }
};