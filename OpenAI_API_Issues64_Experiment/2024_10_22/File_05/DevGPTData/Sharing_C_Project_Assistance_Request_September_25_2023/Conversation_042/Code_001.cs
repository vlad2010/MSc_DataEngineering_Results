// Inside the loop that processes round files
foreach (string currentRoundFilePath in roundFiles)
{
    // Create a new FootballProcessor instance for each round
    FootballProcessor processor = new FootballProcessor(teams, leagueSetup);

    processor.ProcessRoundResults(currentRoundFilePath);
    Console.WriteLine($"Matches in {Path.GetFileName(currentRoundFilePath)} have been processed.");
}
