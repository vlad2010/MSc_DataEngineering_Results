struct AppState
{
    std::vector<float> PlotData = {0.15f, 0.30f, 0.2f, 0.05f}; // existing data for ImPlot pie chart

    // Flags that set whether we show help strings
    bool ShowAssetsInfo = false;
    bool ShowMarkdownInfo = false;
    bool ShowImplotInfo = false;

    char MarkdownInput[512] = "# Welcome to the interactive markdown demo!\nTry writing some markdown content here.";  // new member
};