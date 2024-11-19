if (!processedRoundFiles.Contains(currentRoundFilePath))
{
    processor.ProcessRoundResults(currentRoundFilePath);
    processedRoundFiles.Add(currentRoundFilePath);
    Console.WriteLine($"Matches in {Path.GetFileName(currentRoundFilePath)} have been processed.");
}
else
{
    Console.WriteLine($"Round file {Path.GetFileName(currentRoundFilePath)} has already been processed. Skipping...");
}
