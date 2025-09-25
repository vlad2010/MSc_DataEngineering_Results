using System;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Cryptography.KeyDerivation;

public static class PasswordEncryption
{
    private const int SALT_SIZE = 128 / 8; // 128 bit salt
    private const int HASH_SIZE = 256 / 8; // 256 bit hash
    private const int ITERATIONS = 100000;  // PBKDF2 iterations (NIST recommendation minimum)
    
    public static string HashPassword(string password)
    {
        if (string.IsNullOrEmpty(password))
            throw new ArgumentException("Password cannot be null or empty", nameof(password));
        
        // Generate a random salt
        byte[] salt = new byte[SALT_SIZE];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(salt);
        }
        
        // Hash the password with PBKDF2
        byte[] hash = KeyDerivation.Pbkdf2(
            password: password,
            salt: salt,
            prf: KeyDerivationPrf.HMACSHA256,
            iterationCount: ITERATIONS,
            numBytesRequested: HASH_SIZE
        );
        
        // Combine salt and hash for storage
        byte[] hashBytes = new byte[SALT_SIZE + HASH_SIZE];
        Array.Copy(salt, 0, hashBytes, 0, SALT_SIZE);
        Array.Copy(hash, 0, hashBytes, SALT_SIZE, HASH_SIZE);
        
        // Return base64 encoded result
        return Convert.ToBase64String(hashBytes);
    }
    
    public static bool VerifyPassword(string password, string hashedPassword)
    {
        if (string.IsNullOrEmpty(password) || string.IsNullOrEmpty(hashedPassword))
            return false;
        
        try
        {
            // Extract bytes from stored hash
            byte[] hashBytes = Convert.FromBase64String(hashedPassword);
            
            if (hashBytes.Length != SALT_SIZE + HASH_SIZE)
                return false;
            
            // Extract salt from the stored hash
            byte[] salt = new byte[SALT_SIZE];
            Array.Copy(hashBytes, 0, salt, 0, SALT_SIZE);
            
            // Extract the hash from the stored hash
            byte[] storedHash = new byte[HASH_SIZE];
            Array.Copy(hashBytes, SALT_SIZE, storedHash, 0, HASH_SIZE);
            
            // Compute hash of provided password
            byte[] computedHash = KeyDerivation.Pbkdf2(
                password: password,
                salt: salt,
                prf: KeyDerivationPrf.HMACSHA256,
                iterationCount: ITERATIONS,
                numBytesRequested: HASH_SIZE
            );
            
            // Compare hashes using constant-time comparison
            return CryptographicOperations.FixedTimeEquals(storedHash, computedHash);
        }
        catch
        {
            return false;
        }
    }
    
    // Alternative implementation using BCrypt.Net-Next NuGet package (recommended)
    // Install-Package BCrypt.Net-Next
    /*
    public static class PasswordEncryptionBCrypt
    {
        private const int WORK_FACTOR = 12; // Adjust based on security requirements
        
        public static string HashPassword(string password)
        {
            if (string.IsNullOrEmpty(password))
                throw new ArgumentException("Password cannot be null or empty", nameof(password));
                
            return BCrypt.Net.BCrypt.HashPassword(password, WORK_FACTOR);
        }
        
        public static bool VerifyPassword(string password, string hashedPassword)
        {
            if (string.IsNullOrEmpty(password) || string.IsNullOrEmpty(hashedPassword))
                return false;
                
            try
            {
                return BCrypt.Net.BCrypt.Verify(password, hashedPassword);
            }
            catch
            {
                return false;
            }
        }
    }
    */
}