// Define a route to handle GET requests for fetching users
[HttpGet]
[Route("api/users")]
public IHttpActionResult GetUsers()
{
    // Retrieve user data from a database or any other data source
    var users = ...; // Retrieve users from the backend

    // Return the user data as JSON response
    return Json(users);
}

// Define a route to handle POST requests for creating a new user
[HttpPost]
[Route("api/users")]
public IHttpActionResult CreateUser(UserDto userDto)
{
    // Process the received user data and save it to the backend
    var user = ...; // Process and save the user to the backend

    // Return the created user as JSON response
    return Json(user);
}
