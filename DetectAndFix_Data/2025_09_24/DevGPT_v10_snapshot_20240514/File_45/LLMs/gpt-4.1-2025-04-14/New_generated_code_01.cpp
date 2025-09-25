struct AppState
{
    std::vector<float> PlotData = {0.15f, 0.30f, 0.2f, 0.05f};
    char MarkdownInput[4000] = "*Welcome to the interactive markdown demo!* Try writing some markdown content here.";
    bool ShowAssetsInfo = false;
    bool ShowMarkdownInfo = false;
    bool ShowImplotInfo = false;
};