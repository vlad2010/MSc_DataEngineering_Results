public void ProcessRoundResults(string roundFilePath)
{
    // Read the CSV file and store the data in a List of strings
    List<string> lines = new List<string>();
    using (StreamReader reader = new StreamReader(roundFilePath))
    {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            lines.Add(line);
        }
    }

    // Check if the CSV file has a header row and skip it if it does
    int startIndex = lines[0].StartsWith("HomeTeam,") ? 1 : 0;

    Random random = new Random();

    for (int i = startIndex; i < lines.Count; i++)
    {
        string[] parts = lines[i].Split(',');
        string homeTeam = parts[0];
        string awayTeam = parts[1];
        
        // Generate random scores between 0 and 5 for home and away teams
        int homeGoals = random.Next(0, 6);
        int awayGoals = random.Next(0, 6);
        
        // Create the score string
        string score = $"{homeGoals}-{awayGoals}";
        
        // Update the line with random scores
        parts[2] = score;
        lines[i] = string.Join(",", parts);
        
        // Update team statistics (you need to implement this logic)
        // You should update team statistics based on the random scores here.
        // You can call a method to handle this part.
        UpdateTeamStatistics(homeTeam, awayTeam, homeGoals, awayGoals);
    }

    // Write the updated lines back to the CSV file
    using (StreamWriter writer = new StreamWriter(roundFilePath))
    {
        foreach (string line in lines)
        {
            writer.WriteLine(line);
        }
    }
}
