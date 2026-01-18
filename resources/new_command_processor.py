import subprocess
import argparse
import sys
import threading

def execute_commands(file_path, executable, interactive=False, input_stream=None, silent=False):
    """
    Starts a process and sends commands to its stdin from a text file or piped input.
    Optionally switches to interactive mode after all commands are sent.
    """
    if silent:
        print("Silent mode enabled, sending commands to engine")
    try:
        # Read commands from file or stdin
        if input_stream:
            commands = input_stream.readlines()
        else:
            with open(file_path, 'r') as file:
                commands = file.readlines()

        # Start the engine subprocess
        process = subprocess.Popen(
            executable,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for a "bestmove" response from the engine
        def wait_for_bestmove():
            while True:
                output = process.stdout.readline()
                if output:
                    if not silent:
                        print(f"Process output: {output.strip()}")
                    if output.startswith("bestmove"):
                        break

        # Send each command and optionally wait
        for command in commands:
            command = command.strip()
            if command:
                if not silent:
                    print(f"Sending command: {command}")
                process.stdin.write(command + '\n')
                process.stdin.flush()

                if command == "wait":
                    if not silent:
                        print("Waiting for 'bestmove' response...")
                    wait_for_bestmove()
        # After scripted commands, either go interactive or exit
        if interactive:
            print("\n--- All scripted commands sent. Entering interactive mode ---")
            print("Type UCI commands below. Press Ctrl+D (EOF) to exit.\n")

            # Thread to continuously print engine output
            def stream_output():
                for line in process.stdout:
                    print(f"{line}", end='')

            threading.Thread(target=stream_output, daemon=True).start()

            # Read user input from actual terminal, not from piped stdin
            try:
                with open('/dev/tty', 'r') as tty:
                    while True:
                        try:
                            line = tty.readline()
                        except EOFError:
                            break
                        process.stdin.write(line)
                        process.stdin.flush()
            except FileNotFoundError:
                print("Interactive mode failed: /dev/tty not available.")

        stdout, stderr = process.communicate()
        if not interactive:
            if stdout:
                print("Output:\n" + stdout)
            if stderr:
                print("Errors:\n" + stderr)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send scripted UCI commands, with optional interactive mode.")
    parser.add_argument("executable", type=str, help="Path to the UCI engine executable.")
    parser.add_argument("file_path", nargs="?", type=str, help="Command file. If omitted, stdin is used.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enter interactive mode after scripted commands.")
    parser.add_argument("-s", "--silent", action="store_true", help="Don't print the commands being sent nor their output.")


    args = parser.parse_args()
    input_stream = sys.stdin if not args.file_path else None
    execute_commands(args.file_path, args.executable, interactive=args.interactive, input_stream=input_stream, silent=args.silent)