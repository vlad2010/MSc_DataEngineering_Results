using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

public static class SecureFileGenerator
{
    private static readonly string _rootDir = @"C:\SafeRoot"; // Example root directory

    // Secure random number generator for security-sensitive randomness
    private static int SecureRandomInt(int minValue, int maxValue)
    {
        if (minValue >= maxValue) throw new ArgumentOutOfRangeException();
        byte[] uint32Buffer = new byte[4];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(uint32Buffer);
            uint rand = BitConverter.ToUInt32(uint32Buffer, 0);
            return (int)(minValue + (rand % (uint)(maxValue - minValue)));
        }
    }

    // Sanitize and validate filePath to prevent path traversal
    private static string GetSafeDirectory(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("filePath is required.");

        // Disallow path traversal
        if (filePath.Contains("..") || filePath.Contains(":") || filePath.Contains("/") || filePath.Contains("\\"))
            throw new ArgumentException("Invalid filePath.");

        string fullPath = Path.GetFullPath(Path.Combine(_rootDir, filePath));
        if (!fullPath.StartsWith(_rootDir, StringComparison.OrdinalIgnoreCase))
            throw new UnauthorizedAccessException("Attempted path traversal detected.");

        // Ensure directory exists
        Directory.CreateDirectory(fullPath);
        return fullPath;
    }

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

        int maxRounds = Math.Min(upper.Count, lower.Count) / 2;

        string safeDir = GetSafeDirectory(filePath);

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
                    $"{upper[j % upper.Count]},{SecureRandomInt(lowerGoals, upperGoals)},{upper[(j + i) % upper.Count]},{SecureRandomInt(lowerGoals, upperGoals)}");
            }
            for (int j = 0; j < halfLowerCount; j++)
            {
                csvContent1.AppendLine(
                    $"{lower[j % lower.Count]},{SecureRandomInt(lowerGoals, upperGoals)},{lower[(j + i) % lower.Count]},{SecureRandomInt(lowerGoals, upperGoals)}");
            }

            string fileName1 = $"round-{fileCount}.csv";
            string fullPath1 = Path.Combine(safeDir, fileName1);
            File.WriteAllText(fullPath1, csvContent1.ToString());
            fileCount++;

            // Creating second file
            StringBuilder csvContent2 = new StringBuilder();
            csvContent2.AppendLine("home,home goals,away,away goals");
            for (int j = halfUpperCount; j < upper.Count; j++)
            {
                csvContent2.AppendLine(
                    $"{upper[j % upper.Count]},{SecureRandomInt(lowerGoals, upperGoals)},{upper[(j + i) % upper.Count]},{SecureRandomInt(lowerGoals, upperGoals)}");
            }
            for (int j = halfLowerCount; j < lower.Count; j++)
            {
                csvContent2.AppendLine(
                    $"{lower[j % lower.Count]},{SecureRandomInt(lowerGoals, upperGoals)},{lower[(j + i) % lower.Count]},{SecureRandomInt(lowerGoals, upperGoals)}");
            }

            string fileName2 = $"round-{fileCount}.csv";
            string fullPath2 = Path.Combine(safeDir, fileName2);
            File.WriteAllText(fullPath2, csvContent2.ToString());
            fileCount++;
        }
    }
}