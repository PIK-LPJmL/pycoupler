import subprocess
import socket

# Start the C program as a separate process
@pytest.fixture(scope="session", autouse=True)
def setup_c_program(request):
    # Start the C program using subprocess.Popen
    c_program_process = subprocess.Popen(
        ["./c_program", "arg1", "arg2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the C program to start up, if necessary
    # ... (add any necessary wait logic here)
    
    # Yield the C program process, so that it can be used in the tests
    yield c_program_process
    
    # Clean up after the tests
    c_program_process.terminate()
    c_program_process.wait()

# Test the Python interface's interaction with the C program
def test_python_interface_with_c_program(setup_c_program):
    # Connect to the C program from the Python interface
    c_program_process = setup_c_program
    c_program_port = 12345  # The port number the C program is listening on
    python_interface_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    python_interface_socket.connect(('localhost', c_program_port))

    # Send data from the Python interface to the C program
    python_interface_socket.sendall(b'Hello, C program!')

    # Receive data from the C program in the Python interface
    response = python_interface_socket.recv(1024)
    
    # Assert that the response from the C program is as expected
    assert response == b'Response from C program'
    
    # Clean up after the test
    python_interface_socket.close()