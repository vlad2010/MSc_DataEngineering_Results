using System;
using System.Web.Http;
using System.ComponentModel.DataAnnotations;
using Newtonsoft.Json;

// Define a route to handle GET requests for fetching users
[Authorize] // Ensure that only authenticated users can access this endpoint
[HttpGet]
[Route("api/users")]
public IHttpActionResult GetUsers()
{
    try
    {
        // Retrieve user data from a database or any other data source
        var users = GetUserListFromDatabase(); // Implement data retrieval with appropriate security measures
        
        // Ensure proper encoding if manual encoding is needed
        return Json(users, new JsonSerializerSettings
        {
            // Customize settings if needed for security measures, e.g., not allowing references.
        });
    }
    catch (Exception ex)
    {
        // Log the exception and return a generic error message to the client
        // Do not expose 'ex' details directly
        LogException(ex);
        return InternalServerError(new Exception("An error occurred while processing your request."));
    }
}

// Define a route to handle POST requests for creating a new user
[Authorize] // Ensure that only authenticated users can access this endpoint
[HttpPost]
[Route("api/users")]
public IHttpActionResult CreateUser([FromBody] UserDto userDto)
{
    if (userDto == null)
    {
        return BadRequest("User data must be provided.");
    }

    // Validate user input
    if (!TryValidateModel(userDto))
    {
        return BadRequest("Invalid user data received.");
    }

    // Further input validation/sanitization
    var sanitizedUserDto = SanitizeUserDto(userDto); // Define this function to sanitize input for your environment

    try
    {
        // Process and save the sanitized user data to the backend
        var user = SaveUserToDatabase(sanitizedUserDto); // Implement secure user data saving practices

        // Return the created user as JSON response
        return Json(user, new JsonSerializerSettings
        {
            // Customize settings if needed for security measures, e.g., not allowing references.
        });
    }
    catch (Exception ex)
    {
        // Log the exception and return a generic error message to the client
        LogException(ex);
        return InternalServerError(new Exception("An error occurred while processing your request."));
    }
}

// Dummy methods for illustration purposes. Implementations should handle security.
private List<User> GetUserListFromDatabase()
{
    // Add secure retrieval logic.
    return new List<User>();
}

private User SaveUserToDatabase(UserDto userDto)
{
    // Add secure save logic.
    return new User();
}

private UserDto SanitizeUserDto(UserDto userDto)
{
    // Implement sanitization logic suitable for your environment.
    return userDto;
}

private void LogException(Exception ex)
{
    // Implement logging mechanism.
}

public class UserDto
{
    [Required]
    public string Name { get; set; }
    // Add additional validation attributes as needed
}

public class User
{
    // Define user properties
}