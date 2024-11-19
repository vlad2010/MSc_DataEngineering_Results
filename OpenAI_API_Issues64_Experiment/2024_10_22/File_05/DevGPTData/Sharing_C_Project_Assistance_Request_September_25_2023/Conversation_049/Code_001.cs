using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

class Program
{
    // Function to generate a random score
    static string GenerateRandomScore()
    {
        Random random = new Random();
        int homeGoals = random.Next(0, 6); // Generate a random number of goals for the home team (0-5)
        int awayGoals = random.Next(0, 6); // Generate a random number of goals for the away team (0-5)
        return $"{homeGoals}-{awayGoals}";
    }

    static void Main(string[] args)
    {
        List<string[]> matches = new List<string[]>
        {
            new string[] { "FCK", "BIF", "2023-09-30", "Stadium 1" },
            new string[] { "FCM", "OB", "2023-09-30", "Stadium 2" },
            new string[] { "AaB", "EfB", "2023-09-30", "Stadium 3" },
            new string[] { "SIF", "RFC", "2023-09-30", "Stadium 4" },
            new string[] { "VB", "AGF", "2023-09-30", "Stadium 5" },
            new string[] { "HOB", "LBK", "2023-09-30", "Stadium 6" }
        };

        // Create a StringBuilder to store the CSV data
        StringBuilder csvData = new StringBuilder();
        csvData.AppendLine("HomeTeam,AwayTeam,Score,MatchDate,Stadium");

        foreach (var match in matches)
        {
            string score = GenerateRandomScore();
            string matchDate = match[2];
            string stadium = match[3];
            csvData.AppendLine($"{match[0]},{match[1]},{score},{matchDate},{stadium}");
        }

        // Write the CSV data to a file
        string csvFileName = "random_scores.csv";
        File.WriteAllText(csvFileName, csvData.ToString());

        Console.WriteLine($"Random scores generated and saved to {csvFileName}");
    }
}
