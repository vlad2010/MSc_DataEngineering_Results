using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
    {
        List<string> teams = new List<string>
        {
            "FCK", "BIF", "FCM", "AaB", "OB", "EfB", "SIF", "RFC", "VB", "AGF", "HOB"
        };

        string outputDirectory = @"C:\Your\Output\Directory\"; // Change this to your desired directory

        List<List<string>> rounds = GenerateRoundRobinSchedule(teams, 22);

        for (int roundNumber = 0; roundNumber < rounds.Count; roundNumber++)
        {
            string roundFileName = $"round-{roundNumber + 1}.csv";
            string filePath = Path.Combine(outputDirectory, roundFileName);
            WriteRoundToCsv(filePath, rounds[roundNumber]);
        }

        Console.WriteLine("CSV files saved to the specified directory.");
    }

    // Rest of the code remains the same
    // ...
}
