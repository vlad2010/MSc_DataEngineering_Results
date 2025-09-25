using System;
using System.Collections.Generic;
using System.Linq;
using System.Web.Http;
using System.ComponentModel.DataAnnotations;

// DTO for output (safe fields only)
public class UserOutputDto
{
    public int Id { get; set; }
    public string Username { get; set; }
    // Add other non-sensitive fields as needed
}

// DTO for input (creation)
public class UserInputDto
{
    [Required]
    [StringLength(50, MinimumLength = 3)]
    public string Username { get; set; }

    [Required]
    [StringLength(100, MinimumLength = 6)]
    public string Password { get; set; }
    // Add other fields with validation attributes as needed
}

[Authorize] // Require authentication for all actions
public class UsersController : ApiController
{
    // GET: api/users
    [HttpGet]
    [Route("api/users")]
    public IHttpActionResult GetUsers()
    {
        try
        {
            // Retrieve users from the backend (example)
            var users = GetUsersFromDatabase();

            // Map to output DTO to avoid exposing sensitive data
            var userDtos = users.Select(u => new UserOutputDto
            {
                Id = u.Id,
                Username = u.Username
                // Map other safe fields
            }).ToList();

            return Ok(userDtos);
        }
        catch (Exception)
        {
            // Log exception details internally
            return InternalServerError(); // Do not leak details to client
        }
    }

    // POST: api/users
    [HttpPost]
    [Route("api/users")]
    public IHttpActionResult CreateUser([FromBody] UserInputDto userDto)
    {
        if (!ModelState.IsValid)
        {
            return BadRequest(ModelState);
        }

        try
        {
            // Sanitize and process input
            var sanitizedUsername = userDto.Username.Trim();

            // Hash the password before storing (never store plain text passwords)
            var hashedPassword = HashPassword(userDto.Password);

            // Create user entity (do not allow client to set fields like Id, IsAdmin, etc.)
            var user = new User
            {
                Username = sanitizedUsername,
                PasswordHash = hashedPassword
                // Set other fields as needed
            };

            // Save to backend
            SaveUserToDatabase(user);

            // Return safe output DTO
            var outputDto = new UserOutputDto
            {
                Id = user.Id,
                Username = user.Username
            };

            return CreatedAtRoute("GetUserById", new { id = user.Id }, outputDto);
        }
        catch (Exception)
        {
            // Log exception details internally
            return InternalServerError();
        }
    }

    // Example methods for demonstration (implement as needed)
    private List<User> GetUsersFromDatabase() => new List<User>();
    private void SaveUserToDatabase(User user) { /* ... */ }
    private string HashPassword(string password) => /* Use a secure hash function */ "";

    // User entity (for demonstration)
    private class User
    {
        public int Id { get; set; }
        public string Username { get; set; }
        public string PasswordHash { get; set; }
        // Other fields
    }
}