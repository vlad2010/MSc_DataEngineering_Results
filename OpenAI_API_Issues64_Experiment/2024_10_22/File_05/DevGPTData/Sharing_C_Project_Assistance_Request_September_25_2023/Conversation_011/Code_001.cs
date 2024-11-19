// Inside your Main method after loading setup and team data
for (int roundNumber = 1; roundNumber <= 32; roundNumber++)
{
    string roundFileName = $"round-{roundNumber}.csv";
    string roundFilePath = Path.Combine("Data", roundFileName);

    if (File.Exists(roundFilePath))
    {
        processor.ProcessRoundResults(roundFilePath);
        Console.WriteLine($"Round {roundNumber} matches have been processed.");
    }
    else
    {
        Console.WriteLine($"Round {roundNumber} file not found. Please check the file name.");
    }
}
