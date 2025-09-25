using System;
using System.Security.Cryptography;
using System.Text;

public static class PasswordHashing
{
    // Hash a password with a random salt using PBKDF2
    public static string HashPassword(string password)
    {
        // Generate a 16-byte salt
        byte[] salt = new byte[16];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(salt);
        }

        // Derive a 32-byte subkey (hash) using PBKDF2 with 100,000 iterations
        using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, 100_000, HashAlgorithmName.SHA256))
        {
            byte[] hash = pbkdf2.GetBytes(32);

            // Combine salt and hash for storage (Base64-encoded)
            string saltBase64 = Convert.ToBase64String(salt);
            string hashBase64 = Convert.ToBase64String(hash);
            return $"{saltBase64}:{hashBase64}";
        }
    }

    // Verify a password against a stored salt:hash string
    public static bool VerifyPassword(string password, string stored)
    {
        var parts = stored.Split(':');
        if (parts.Length != 2)
            return false;

        byte[] salt = Convert.FromBase64String(parts[0]);
        byte[] storedHash = Convert.FromBase64String(parts[1]);

        using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, 100_000, HashAlgorithmName.SHA256))
        {
            byte[] hash = pbkdf2.GetBytes(32);
            // Use constant-time comparison to prevent timing attacks
            return CryptographicOperations.FixedTimeEquals(hash, storedHash);
        }
    }
}