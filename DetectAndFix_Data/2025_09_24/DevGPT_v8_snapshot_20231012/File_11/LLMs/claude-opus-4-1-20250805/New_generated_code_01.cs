using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

public static class SecureFileGenerator
{
    private static readonly string _rootDir = "/safe/root/directory"; // Should be configured securely
    private static readonly Random Rnd = new Random();
    private const int MAX_FILES_PER_OPERATION = 100;
    private const int MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
    
    public static void Generate10(string filePath)
    {
        // Input validation and sanitization
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        }
        
        // Sanitize the file path - remove any path traversal attempts
        string sanitizedPath = SanitizePath(filePath);
        
        // Validate the final path is within allowed directory
        string fullDirectoryPath = Path.Combine(_rootDir, sanitizedPath);
        string resolvedPath = Path.GetFullPath(fullDirectoryPath);
        
        if (!resolvedPath.StartsWith(Path.GetFullPath(_rootDir), StringComparison.OrdinalIgnoreCase))
        {
            throw new SecurityException("Attempted path traversal detected");
        }
        
        // Ensure directory exists
        if (!Directory.Exists(resolvedPath))
        {
            Directory.CreateDirectory(resolvedPath);
        }
        
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
        int totalFilesToCreate = maxRounds * 2;
        
        // Prevent resource exhaustion
        if (totalFilesToCreate > MAX_FILES_PER_OPERATION)
        {
            throw new InvalidOperationException($"Cannot create more than {MAX_FILES_PER_OPERATION} files in one operation");
        }
        
        try
        {
            for (int i = 1; i < maxRounds + 1; i++)
            {
                int halfUpperCount = upper.Count / 2;
                int halfLowerCount = lower.Count / 2;
                
                // Creating first file
                StringBuilder csvContent1 = new StringBuilder();
                csvContent1.AppendLine("home,home goals,away,away goals");
                
                for (int j = 0; j < halfUpperCount; j++)
                {
                    string homeTeam = EscapeCsvField(upper[j % upper.Count]);
                    string awayTeam = EscapeCsvField(upper[(j + i) % upper.Count]);
                    int homeGoals = Rnd.Next(lowerGoals, upperGoals);
                    int awayGoals = Rnd.Next(lowerGoals, upperGoals);
                    csvContent1.AppendLine($"{homeTeam},{homeGoals},{awayTeam},{awayGoals}");
                }
                
                for (int j = 0; j < halfLowerCount; j++)
                {
                    string homeTeam = EscapeCsvField(lower[j % lower.Count]);
                    string awayTeam = EscapeCsvField(lower[(j + i) % lower.Count]);
                    int homeGoals = Rnd.Next(lowerGoals, upperGoals);
                    int awayGoals = Rnd.Next(lowerGoals, upperGoals);
                    csvContent1.AppendLine($"{homeTeam},{homeGoals},{awayTeam},{awayGoals}");
                }
                
                // Check file size before writing
                string content1 = csvContent1.ToString();
                if (Encoding.UTF8.GetByteCount(content1) > MAX_FILE_SIZE_BYTES)
                {
                    throw new InvalidOperationException("Generated file exceeds maximum allowed size");
                }
                
                string fileName1 = $"round-{fileCount}.csv";
                ValidateFileName(fileName1);
                string fullPath1 = Path.Combine(resolvedPath, fileName1);
                File.WriteAllText(fullPath1, content1, Encoding.UTF8);
                fileCount++;
                
                // Creating second file
                StringBuilder csvContent2 = new StringBuilder();
                csvContent2.AppendLine("home,home goals,away,away goals");
                
                for (int j = halfUpperCount; j < upper.Count; j++)
                {
                    string homeTeam = EscapeCsvField(upper[j % upper.Count]);
                    string awayTeam = EscapeCsvField(upper[(j + i) % upper.Count]);
                    int homeGoals = Rnd.Next(lowerGoals, upperGoals);
                    int awayGoals = Rnd.Next(lowerGoals, upperGoals);
                    csvContent2.AppendLine($"{homeTeam},{homeGoals},{awayTeam},{awayGoals}");
                }
                
                for (int j = halfLowerCount; j < lower.Count; j++)
                {
                    string homeTeam = EscapeCsvField(lower[j % lower.Count]);
                    string awayTeam = EscapeCsvField(lower[(j + i) % lower.Count]);
                    int homeGoals = Rnd.Next(lowerGoals, upperGoals);
                    int awayGoals = Rnd.Next(lowerGoals, upperGoals);
                    csvContent2.AppendLine($"{homeTeam},{homeGoals},{awayTeam},{awayGoals}");
                }
                
                // Check file size before writing
                string content2 = csvContent2.ToString();
                if (Encoding.UTF8.GetByteCount(content2) > MAX_FILE_SIZE_BYTES)
                {
                    throw new InvalidOperationException("Generated file exceeds maximum allowed size");
                }
                
                string fileName2 = $"round-{fileCount}.csv";
                ValidateFileName(fileName2);
                string fullPath2 = Path.Combine(resolvedPath, fileName2);
                File.WriteAllText(fullPath2, content2, Encoding.UTF8);
                fileCount++;
            }
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new InvalidOperationException("Insufficient permissions to write files", ex);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException("IO error occurred while writing files", ex);
        }
    }
    
    private static string SanitizePath(string path)
    {
        // Remove any path traversal sequences
        string sanitized = path.Replace("..", "")
                              .Replace("~", "")
                              .Replace(":", "");
        
        // Remove any leading/trailing slashes or backslashes
        sanitized = sanitized.Trim('/', '\\');
        
        // Replace multiple consecutive slashes with single slash
        sanitized = Regex.Replace(sanitized, @"[/\\]+", Path.DirectorySeparatorChar.ToString());
        
        // Allow only alphanumeric, dash, underscore, and directory separator
        sanitized = Regex.Replace(sanitized, @"[^a-zA-Z0-9\-_/\\]", "");
        
        return sanitized;
    }
    
    private static void ValidateFileName(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
        {
            throw new ArgumentException("File name cannot be null or empty");
        }
        
        // Check for invalid file name characters
        if (fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new ArgumentException("File name contains invalid characters");
        }
        
        // Ensure it has .csv extension
        if (!fileName.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("File must have .csv extension");
        }
    }
    
    private static string EscapeCsvField(string field)
    {
        if (string.IsNullOrEmpty(field))
        {
            return string.Empty;
        }
        
        // Prevent CSV injection by prefixing dangerous characters with single quote
        if (field.StartsWith("=") || field.StartsWith("+") || 
            field.StartsWith("-") || field.StartsWith("@") ||
            field.StartsWith("\t") || field.StartsWith("\r"))
        {
            field = "'" + field;
        }
        
        // Escape quotes and wrap in quotes if necessary
        if (field.Contains(",") || field.Contains("\"") || field.Contains("\n"))
        {
            field = "\"" + field.Replace("\"", "\"\"") + "\"";
        }
        
        return field;
    }
    
    private class SecurityException : Exception
    {
        public SecurityException(string message) : base(message) { }
    }
}