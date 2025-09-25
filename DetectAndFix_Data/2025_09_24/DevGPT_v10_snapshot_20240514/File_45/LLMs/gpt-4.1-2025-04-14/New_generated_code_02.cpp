#include <vector>
#include <string>
#include <algorithm>

class AppState
{
public:
    AppState()
        : PlotData{0.15f, 0.30f, 0.2f, 0.05f},
          MarkdownInput("*Welcome to the interactive markdown demo!* Try writing some markdown content here."),
          ShowAssetsInfo(false),
          ShowMarkdownInfo(false),
          ShowImplotInfo(false)
    {}

    // Safe setter for MarkdownInput with length check
    void SetMarkdownInput(const std::string& input) {
        // Limit input to 3999 characters (if you want to keep a similar limit)
        if (input.size() > MaxMarkdownInputLength) {
            MarkdownInput = input.substr(0, MaxMarkdownInputLength);
        } else {
            MarkdownInput = input;
        }
    }

    const std::string& GetMarkdownInput() const {
        return MarkdownInput;
    }

    // Other getters/setters as needed...

private:
    static constexpr size_t MaxMarkdownInputLength = 3999;
    std::vector<float> PlotData;
    std::string MarkdownInput;
    bool ShowAssetsInfo;
    bool ShowMarkdownInfo;
    bool ShowImplotInfo;
};