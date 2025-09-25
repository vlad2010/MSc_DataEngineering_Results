#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

struct AppState
{
    // Use std::vector for dynamic sizing instead of raw float array
    std::vector<float> PlotData = {0.15f, 0.30f, 0.2f, 0.05f};
    
    // Option 1: Use std::string for safer string handling (Recommended)
    std::string MarkdownInput = "*Welcome to the interactive markdown demo!* Try writing some markdown content here.";
    
    // Option 2: If you must use char array for compatibility (e.g., ImGui), 
    // use safer practices with explicit size constants
    static constexpr size_t MARKDOWN_BUFFER_SIZE = 4000;
    char MarkdownInputBuffer[MARKDOWN_BUFFER_SIZE] = {};  // Zero-initialized
    
    bool ShowAssetsInfo = false;
    bool ShowMarkdownInfo = false;
    bool ShowImplotInfo = false;
    
    // Constructor to ensure proper initialization
    AppState() {
        // If using char buffer, safely copy initial content
        const char* initialText = "*Welcome to the interactive markdown demo!* Try writing some markdown content here.";
        strncpy(MarkdownInputBuffer, initialText, MARKDOWN_BUFFER_SIZE - 1);
        MarkdownInputBuffer[MARKDOWN_BUFFER_SIZE - 1] = '\0';  // Ensure null termination
    }
    
    // Safe setter method for markdown input (if using char buffer)
    bool SetMarkdownInput(const char* input) {
        if (!input) return false;
        
        size_t inputLen = strlen(input);
        if (inputLen >= MARKDOWN_BUFFER_SIZE) {
            // Input too large, truncate or reject
            return false;
        }
        
        strncpy(MarkdownInputBuffer, input, MARKDOWN_BUFFER_SIZE - 1);
        MarkdownInputBuffer[MARKDOWN_BUFFER_SIZE - 1] = '\0';
        return true;
    }
    
    // Safe getter method
    const char* GetMarkdownInput() const {
        return MarkdownInputBuffer;
    }
    
    // If using with ImGui, provide a safe interface
    void GetMarkdownBufferForImGui(char** buffer, size_t* size) {
        *buffer = MarkdownInputBuffer;
        *size = MARKDOWN_BUFFER_SIZE;
    }
    
    // Method to validate plot data (prevent invalid float values)
    bool ValidatePlotData() {
        return std::all_of(PlotData.begin(), PlotData.end(), 
            [](float val) { 
                return std::isfinite(val) && val >= 0.0f && val <= 1.0f; 
            });
    }
    
    // Sanitize markdown input to prevent XSS if rendered in web context
    void SanitizeMarkdownInput() {
        if (!MarkdownInput.empty()) {
            // Basic sanitization - extend based on your markdown parser
            std::string& str = MarkdownInput;
            size_t pos = 0;
            
            // Remove potential script tags
            while ((pos = str.find("<script", pos)) != std::string::npos) {
                str.replace(pos, 7, "&lt;script");
                pos += 10;
            }
            
            // Remove potential event handlers
            const std::vector<std::string> dangerous = {"onclick", "onerror", "onload"};
            for (const auto& pattern : dangerous) {
                pos = 0;
                while ((pos = str.find(pattern, pos)) != std::string::npos) {
                    str.erase(pos, pattern.length());
                }
            }
        }
    }
};