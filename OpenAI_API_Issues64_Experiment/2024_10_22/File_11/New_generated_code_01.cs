using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

public static void Generate10(string filePath)
{
    List<string> upper = new List<string>
    {
        "FCK", "RFC", "FCM", "HBK", "AGF", "HOB"
    };
    
    List<string> lower = new List<string>
    {
        "AAB", "SIF", "VFF", "EFC", "BIF", "FCN"
    };

    int fileCount = 23;
    const int lowerGoals = 0;
    const int upperGoals = 6;

    // Ensure filePath is validated against path traversal
    // This is a simple example, but it may need to be more comprehensive based on your actual needs
    if (!IsValidPath(filePath))
    {
        throw new ArgumentException("Invalid file path.");
    }

    int maxRounds = Math.Min(upper.Count, lower.Count) / 2;

    byte[] randomBuffer = new byte[4]; // Buffer to store random number bytes
    RandomNumberGenerator rng = RandomNumberGenerator.Create();

    for (int i = 1; i < maxRounds + 1; i++)
    {
        int halfUpperCount = upper.Count / 2;
        int halfLowerCount = lower.Count / 2;
        
        // Creating first file
        StringBuilder csvContent1 = new StringBuilder();
        csvContent1.AppendLine("home,home goals,away,away goals");
        for (int j = 0; j < halfUpperCount; j++)
        {
            csvContent1.AppendLine(
                $"{upper[j % upper.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)},{upper[(j + i) % upper.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)}");
        }
        for (int j = 0; j < halfLowerCount; j++)
        {
            csvContent1.AppendLine(
                $"{lower[j % lower.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)},{lower[(j + i) % lower.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)}");
        }
        
        string fileName1 = $"round-{fileCount}.csv";
        string fullPath1 = Path.Combine(_rootDir, filePath, fileName1);
        File.WriteAllText(fullPath1, csvContent1.ToString());
        fileCount++;
        
        // Creating second file
        StringBuilder csvContent2 = new StringBuilder();
        csvContent2.AppendLine("home,home goals,away,away goals");
        for (int j = halfUpperCount; j < upper.Count; j++)
        {
            csvContent2.AppendLine(
                $"{upper[j % upper.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)},{upper[(j + i) % upper.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)}");
        }
        for (int j = halfLowerCount; j < lower.Count; j++)
        {
            csvContent2.AppendLine(
                $"{lower[j % lower.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)},{lower[(j + i) % lower.Count]},{GetRandomNumber(rng, lowerGoals, upperGoals)}");
        }

        string fileName2 = $"round-{fileCount}.csv";
        string fullPath2 = Path.Combine(_rootDir, filePath, fileName2);
        File.WriteAllText(fullPath2, csvContent2.ToString());
        fileCount++;
    }
    
    rng.Dispose();
}

private static int GetRandomNumber(RandomNumberGenerator rng, int minValue, int maxValue)
{
    if (minValue > maxValue) throw new ArgumentOutOfRangeException(nameof(minValue), $"{nameof(minValue)} cannot be greater than {nameof(maxValue)}.");

    long range = (long) maxValue - minValue;
    if (range <= 0) throw new ArgumentOutOfRangeException(nameof(maxValue));

    // Create a random value in a secure manner
    byte[] randomBytes = new byte[4];
    rng.GetBytes(randomBytes);
    int randomNumber = Math.Abs(BitConverter.ToInt32(randomBytes, 0));
    return (int)(minValue + (randomNumber % range));
}

private static bool IsValidPath(string path)
{
    // A simple method to validate the file path for demonstration purposes
    // Ensure it doesn't attempt directory traversal
    Regex pathPattern = new Regex(@"^([a-zA-Z]:\\)?((\.)|(\\[^\\/:*?""<>|]*)+\\?)$");
    return pathPattern.IsMatch(path) && !path.Contains("..");
}