using System;
using System.Security.Cryptography;
using System.Text;

public static class PasswordEncryption
{
    public static string EncryptPassword(string userName)
    {
        using (SHA256 sha256Hash = SHA256.Create())
        {
            byte[] data = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(userName));
            StringBuilder sBuilder = new StringBuilder();
            for (int i = 0; i < data.Length; i++)
            {
                sBuilder.Append(data[i].ToString("x2"));
            }
            return sBuilder.ToString();
        }
    }
}