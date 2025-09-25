void DemoMarkdown(AppState& appState)
{
    ImGuiMd::RenderUnindented(R"(
        # Demo markdown usage

        *Enter your markdown content below:*
        )"
    );

    ImGui::InputTextMultiline("Markdown Input", appState.MarkdownInput, sizeof(appState.MarkdownInput), ImVec2(-1, HelloImGui::EmToPixels(10)));

    ImGuiMd::RenderUnindented(appState.MarkdownInput);  // render the user's input

    ImGui::Checkbox("More info##Markdown", &appState.ShowMarkdownInfo);
    if (appState.ShowMarkdownInfo)
        ImGuiMd::RenderUnindented(GetDoc("MarkdownDoc"));
}