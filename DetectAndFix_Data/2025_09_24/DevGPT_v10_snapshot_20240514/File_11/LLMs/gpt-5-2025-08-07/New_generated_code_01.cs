using System;
using System.Security.Cryptography;

public static class PasswordEncryption
{
    // PBKDF2 parameters (tune Iterations to your performance/SLA; benchmark on prod hardware)
    private const int SaltSize = 16;         // 128-bit salt
    private const int KeySize = 32;          // 256-bit derived key
    private const int Iterations = 600_000;  // Example modern default (adjust as needed)
    private static readonly HashAlgorithmName Algorithm = HashAlgorithmName.SHA256;

    // New, correct API: create a password hash with salt and iterations
    public static string HashPassword(string password)
    {
        if (password == null) throw new ArgumentNullException(nameof(password));

        byte[] salt = RandomNumberGenerator.GetBytes(SaltSize);
        byte[] hash = Rfc2898DeriveBytes.Pbkdf2(
            password: password,
            salt: salt,
            iterations: Iterations,
            hashAlgorithm: Algorithm,
            derivedKeyLength: KeySize
        );

        // Store as: scheme$alg$iter$saltBase64$hashBase64
        return $"PBKDF2${Algorithm.Name}${Iterations}${Convert.ToBase64String(salt)}${Convert.ToBase64String(hash)}";
    }

    // Verify a password against a stored hash
    public static bool VerifyPassword(string password, string stored)
    {
        if (password == null) throw new ArgumentNullException(nameof(password));
        if (string.IsNullOrWhiteSpace(stored)) return false;

        var parts = stored.Split('$');
        if (parts.Length != 5 || parts[0] != "PBKDF2") return false;

        var alg = new HashAlgorithmName(parts[1]);
        if (!int.TryParse(parts[2], out int iterations) || iterations <= 0) return false;

        byte[] salt, expectedHash;
        try
        {
            salt = Convert.FromBase64String(parts[3]);
            expectedHash = Convert.FromBase64String(parts[4]);
        }
        catch (FormatException)
        {
            return false;
        }

        byte[] actualHash = Rfc2898DeriveBytes.Pbkdf2(
            password: password,
            salt: salt,
            iterations: iterations,
            hashAlgorithm: alg,
            derivedKeyLength: expectedHash.Length
        );

        // Constant-time comparison to mitigate timing attacks (CWE-208)
        return CryptographicOperations.FixedTimeEquals(expectedHash, actualHash);
    }

    // Optional: helper to decide if a hash should be upgraded (e.g., iterations increased)
    public static bool NeedsRehash(string stored)
    {
        var parts = stored?.Split('$');
        if (parts == null || parts.Length != 5 || parts[0] != "PBKDF2") return true;
        if (!int.TryParse(parts[2], out int iterations)) return true;
        return iterations < Iterations || !string.Equals(parts[1], Algorithm.Name, StringComparison.Ordinal);
    }

    // Old method: disable to prevent insecure usage and wrong input (username vs password).
    // This method used SHA-256 without salt/iterations and hashed the username, not the password.
    [Obsolete("Insecure. Use HashPassword() and VerifyPassword() instead.", true)]
    public static string EncryptPassword(string userName) =>
        throw new NotSupportedException("EncryptPassword is insecure and has been disabled. Use HashPassword/VerifyPassword.");
}