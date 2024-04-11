import argparse
import sys
import time

import pkg_resources

from ..core.core import OpenInterpreter
from .conversation_navigator import conversation_navigator
from .profiles.profiles import open_storage_dir, profile, reset_profile
from .utils.check_for_update import check_for_update
from .utils.display_markdown_message import display_markdown_message
from .validate_llm_settings import validate_llm_settings


def start_terminal_interface(interpreter):
    """
    Meant to be used from the command line. Parses arguments, starts OI's terminal interface.
    """

    arguments = [
        {
            "name": "profile",
            "nickname": "p",
            "help_text": "name of profile. run `--profiles` to open profile directory",
            "type": str,
            "default": "default.yaml",
        },
        {
            "name": "custom_instructions",
            "nickname": "ci",
            "help_text": "custom instructions for the language model. will be appended to the system_message",
            "type": str,
            "attribute": {"object": interpreter, "attr_name": "custom_instructions"},
        },
        {
            "name": "system_message",
            "nickname": "s",
            "help_text": "(we don't recommend changing this) base prompt for the language model",
            "type": str,
            "attribute": {"object": interpreter, "attr_name": "system_message"},
        },
        {
            "name": "auto_run",
            "nickname": "y",
            "help_text": "automatically run generated code",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "auto_run"},
        },
        {
            "name": "verbose",
            "nickname": "v",
            "help_text": "print detailed logs",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "verbose"},
        },
        {
            "name": "model",
            "nickname": "m",
            "help_text": "language model to use",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "model"},
        },
        {
            "name": "temperature",
            "nickname": "t",
            "help_text": "optional temperature setting for the language model",
            "type": float,
            "attribute": {"object": interpreter.llm, "attr_name": "temperature"},
        },
        {
            "name": "llm_supports_vision",
            "nickname": "lsv",
            "help_text": "inform OI that your model supports vision, and can recieve vision inputs",
            "type": bool,
            "action": argparse.BooleanOptionalAction,
            "attribute": {"object": interpreter.llm, "attr_name": "supports_vision"},
        },
        {
            "name": "llm_supports_functions",
            "nickname": "lsf",
            "help_text": "inform OI that your model supports OpenAI-style functions, and can make function calls",
            "type": bool,
            "action": argparse.BooleanOptionalAction,
            "attribute": {"object": interpreter.llm, "attr_name": "supports_functions"},
        },
        {
            "name": "context_window",
            "nickname": "cw",
            "help_text": "optional context window size for the language model",
            "type": int,
            "attribute": {"object": interpreter.llm, "attr_name": "context_window"},
        },
        {
            "name": "max_tokens",
            "nickname": "x",
            "help_text": "optional maximum number of tokens for the language model",
            "type": int,
            "attribute": {"object": interpreter.llm, "attr_name": "max_tokens"},
        },
        {
            "name": "max_budget",
            "nickname": "b",
            "help_text": "optionally set the max budget (in USD) for your llm calls",
            "type": float,
            "attribute": {"object": interpreter.llm, "attr_name": "max_budget"},
        },
        {
            "name": "api_base",
            "nickname": "ab",
            "help_text": "optionally set the API base URL for your llm calls (this will override environment variables)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "api_base"},
        },
        {
            "name": "api_key",
            "nickname": "ak",
            "help_text": "optionally set the API key for your llm calls (this will override environment variables)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "api_key"},
        },
        {
            "name": "api_version",
            "nickname": "av",
            "help_text": "optionally set the API version for your llm calls (this will override environment variables)",
            "type": str,
            "attribute": {"object": interpreter.llm, "attr_name": "api_version"},
        },
        {
            "name": "max_output",
            "nickname": "xo",
            "help_text": "optional maximum number of characters for code outputs",
            "type": int,
            "attribute": {"object": interpreter, "attr_name": "max_output"},
        },
        {
            "name": "force_task_completion",
            "nickname": "fc",
            "help_text": "runs OI in a loop, requiring it to admit to completing/failing task",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "force_task_completion"},
        },
        {
            "name": "disable_telemetry",
            "nickname": "dt",
            "help_text": "disables sending of basic anonymous usage stats",
            "type": bool,
            "default": False,
            "attribute": {"object": interpreter, "attr_name": "disable_telemetry"},
        },
        {
            "name": "offline",
            "nickname": "o",
            "help_text": "turns off all online features (except the language model, if it's hosted)",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "offline"},
        },
        {
            "name": "speak_messages",
            "nickname": "sm",
            "help_text": "(Mac only, experimental) use the applescript `say` command to read messages aloud",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "speak_messages"},
        },
        {
            "name": "safe_mode",
            "nickname": "safe",
            "help_text": "optionally enable safety mechanisms like code scanning; valid options are off, ask, and auto",
            "type": str,
            "choices": ["off", "ask", "auto"],
            "default": "off",
            "attribute": {"object": interpreter, "attr_name": "safe_mode"},
        },
        {
            "name": "debug",
            "nickname": "debug",
            "help_text": "debug mode for open interpreter developers",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "debug"},
        },
        {
            "name": "fast",
            "nickname": "f",
            "help_text": "runs `interpreter --model gpt-3.5-turbo` and asks OI to be extremely concise",
            "type": bool,
        },
        {
            "name": "multi_line",
            "nickname": "ml",
            "help_text": "enable multi-line inputs starting and ending with ```",
            "type": bool,
            "attribute": {"object": interpreter, "attr_name": "multi_line"},
        },
        {
            "name": "local",
            "nickname": "l",
            "help_text": "experimentally run the LLM locally via Llamafile (this changes many more settings than `--offline`)",
            "type": bool,
        },
        {
            "name": "vision",
            "nickname": "vi",
            "help_text": "experimentally use vision for supported languages",
            "type": bool,
        },
        {
            "name": "os",
            "nickname": "os",
            "help_text": "experimentally let Open Interpreter control your mouse and keyboard",
            "type": bool,
        },
        # Special commands
        {
            "name": "reset_profile",
            "help_text": "reset a profile file. run `--reset_profile` without an argument to reset all default profiles",
            "type": str,
            "default": "NOT_PROVIDED",
            "nargs": "?",  # This means you can pass in nothing if you want
        },
        {"name": "profiles", "help_text": "opens profiles directory", "type": bool},
        {
            "name": "local_models",
            "help_text": "opens local models directory",
            "type": bool,
        },
        {
            "name": "conversations",
            "help_text": "list conversations to resume",
            "type": bool,
        },
        {
            "name": "server",
            "help_text": "start open interpreter as a server",
            "type": bool,
        },
        {
            "name": "version",
            "help_text": "get Open Interpreter's version number",
            "type": bool,
        },
    ]

    # Check for deprecated flags before parsing arguments
    deprecated_flags = {
        "--debug_mode": "--verbose",
    }

    for old_flag, new_flag in deprecated_flags.items():
        if old_flag in sys.argv:
            print(f"\n`{old_flag}` has been renamed to `{new_flag}`.\n")
            time.sleep(1.5)
            sys.argv.remove(old_flag)
            sys.argv.append(new_flag)

    parser = argparse.ArgumentParser(
        description="Open Interpreter", usage="%(prog)s [options]"
    )

    # Add arguments
    for arg in arguments:
        default = arg.get("default")
        action = arg.get("action", "store_true")
        nickname = arg.get("nickname")

        name_or_flags = [f'--{arg["name"]}']
        if nickname:
            name_or_flags.append(f"-{nickname}")

        # Construct argument name flags
        flags = (
            [f"-{nickname}", f'--{arg["name"]}'] if nickname else [f'--{arg["name"]}']
        )

        if arg["type"] == bool:
            parser.add_argument(
                *flags,
                dest=arg["name"],
                help=arg["help_text"],
                action=action,
                default=default,
            )
        else:
            choices = arg.get("choices")
            parser.add_argument(
                *flags,
                dest=arg["name"],
                help=arg["help_text"],
                type=arg["type"],
                choices=choices,
                default=default,
                nargs=arg.get("nargs"),
            )

    args, unknown_args = parser.parse_known_args()

    # handle unknown arguments
    if unknown_args:
        print(f"\nUnrecognized argument(s): {unknown_args}")
        parser.print_usage()
        print(
            "For detailed documentation of supported arguments, please visit: https://docs.openinterpreter.com/settings/all-settings"
        )
        sys.exit(1)

    if args.profiles:
        open_storage_dir("profiles")
        return

    if args.local_models:
        open_storage_dir("models")
        return

    if args.reset_profile is not None and args.reset_profile != "NOT_PROVIDED":
        reset_profile(
            args.reset_profile
        )  # This will be None if they just ran `--reset_profile`
        return

    if args.version:
        version = pkg_resources.get_distribution("open-interpreter").version
        update_name = "New Computer Update"  # Change this with each major update
        print(f"Open Interpreter {version} {update_name}")
        return

    # if safe_mode and auto_run are enabled, safe_mode disables auto_run
    if interpreter.auto_run and (
        interpreter.safe_mode == "ask" or interpreter.safe_mode == "auto"
    ):
        setattr(interpreter, "auto_run", False)

    if args.fast:
        args.profile = "fast.yaml"

    if args.vision:
        args.profile = "vision.yaml"

    if args.os:
        interpreter.os = True
        interpreter.llm.supports_vision = True
        # interpreter.shrink_images = True # Faster but less accurate

        if not args.model:
            args.model = "gpt-4-vision-preview"

        interpreter.llm.supports_functions = False
        interpreter.llm.context_window = 110000
        interpreter.llm.max_tokens = 4096
        interpreter.auto_run = True
        interpreter.force_task_completion = True

        interpreter.system_message = """
        
You are Open Interpreter, a world-class programmer that can complete any goal by executing code.

When you write code, it will be executed **on the user's machine**. The user has given you **full and complete permission** to execute any code necessary to complete the task.

When a user refers to a filename, they're likely referring to an existing file in the directory you're currently executing code in.

In general, try to make plans with as few steps as possible. As for actually executing code to carry out that plan, **don't try to do everything in one code block.** You should try something, print information about it, then continue from there in tiny, informed steps. You will never get it on the first try, and attempting it in one go will often lead to errors you cant see.

Manually summarize text.

Do not try to write code that attempts the entire task at once, and verify at each step whether or not you're on track.

# Computer

You may use the `interpreter` Python module to complete tasks:

```python
interpreter.interpreter.computer.display.view() # Shows you what's on the screen, returns a `pil_image` `in case you need it (rarely). **You almost always want to do this first!**

interpreter.interpreter.computer.keyboard.write("hello")

interpreter.interpreter.computer.mouse.click("text onscreen") # This clicks on the UI element with that text. Use this **frequently** and get creative! To click a video, you could pass the *timestamp* (which is usually written on the thumbnail) into this.
interpreter.interpreter.computer.mouse.move("open recent >") # This moves the mouse over the UI element with that text. Many dropdowns will disappear if you click them. You have to hover over items to reveal more.
interpreter.interpreter.computer.mouse.click(x=500, y=500) # Use this very, very rarely. It's highly inaccurate
interpreter.interpreter.computer.mouse.click(icon="gear icon") # Moves mouse to the icon with that description. Use this very often

interpreter.interpreter.computer.mouse.scroll(-10) # Scrolls down. If you don't find some text on screen that you expected to be there, you probably want to do this
x, y = interpreter.interpreter.computer.display.center() # Get your bearings

interpreter.interpreter.computer.clipboard.view() # Returns contents of clipboard
interpreter.interpreter.computer.os.get_selected_text() # Use frequently. If editing text, the user often wants this
```

For rare and complex mouse actions, consider using computer vision libraries on the `interpreter.interpreter.computer.display.view()` `pil_image` to produce a list of coordinates for the mouse to move/drag to.

If the user highlighted text in an editor, then asked you to modify it, they probably want you to `interpreter.interpreter.computer.keyboard.write` over their version of the text.

Tasks are 100% computer-based. DO NOT simply write long messages to the user to complete tasks. You MUST put your text back into the program they're using to deliver your text!

Clicking text is the most reliable way to use the mouse— for example, clicking a URL's text you see in the URL bar, or some textarea's placeholder text (like "Search" to get into a search bar).

Applescript might be best for some tasks.

If you use `plt.show()`, the resulting image will be sent to you. However, if you use `PIL.Image.show()`, the resulting image will NOT be sent to you.

It is very important to make sure you are focused on the right application and window. Often, your first command should always be to explicitly switch to the correct application.

When searching the web, use query parameters. For example, https://www.amazon.com/s?k=monitor

Try multiple methods before saying the task is impossible. **You can do it!**

# Critical Routine Procedure for Multi-Step Tasks

Include `interpreter.interpreter.computer.display.view()` after a 2 second delay at the end of _every_ code block to verify your progress, then answer these questions in extreme detail:

1. Generally, what is happening on-screen?
2. What is the active app?
3. What hotkeys does this app support that might get be closer to my goal?
4. What text areas are active, if any?
5. What text is selected?
6. What options could you take next to get closer to your goal?
        
        """.strip()

        if args.offline:
            # Icon finding does not work offline
            interpreter.system_message = interpreter.system_message.replace(
                'interpreter.interpreter.computer.mouse.click(icon="gear icon") # Moves mouse to the icon with that description. Use this very often\n',
                "",
            )

        # Check if required packages are installed

        # THERE IS AN INCONSISTENCY HERE.
        # We should be testing if they import WITHIN OI's computer, not here.

        packages = ["cv2", "plyer", "pyautogui", "pyperclip", "pywinctl"]
        missing_packages = []
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            display_markdown_message(
                f"> **Missing Package(s): {', '.join(['`' + p + '`' for p in missing_packages])}**\n\nThese packages are required for OS Control.\n\nInstall them?\n"
            )
            user_input = input("(y/n) > ")
            if user_input.lower() != "y":
                print("\nPlease try to install them manually.\n\n")
                time.sleep(2)
                print("Attempting to start OS control anyway...\n\n")

            for pip_name in ["pip", "pip3"]:
                command = f"{pip_name} install 'open-interpreter[os]'"
                
                interpreter.computer.run("shell", command, display=True)

                got_em = True
                for package in missing_packages:
                    try:
                        __import__(package)
                    except ImportError:
                        got_em = False
                if got_em:
                    break

            missing_packages = []
            for package in packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            if missing_packages != []:
                print(
                    "\n\nWarning: The following packages could not be installed:",
                    ", ".join(missing_packages),
                )
                print("\nPlease try to install them manually.\n\n")
                time.sleep(2)
                print("Attempting to start OS control anyway...\n\n")

        display_markdown_message("> `OS Control` enabled")

        # Should we explore other options for ^ these kinds of tags?
        # Like:

        # from rich import box
        # from rich.console import Console
        # from rich.panel import Panel
        # console = Console()
        # print(">\n\n")
        # console.print(Panel("[bold italic white on black]OS CONTROL[/bold italic white on black] Enabled", box=box.SQUARE, expand=False), style="white on black")
        # print(">\n\n")
        # console.print(Panel("[bold italic white on black]OS CONTROL[/bold italic white on black] Enabled", box=box.HEAVY, expand=False), style="white on black")
        # print(">\n\n")
        # console.print(Panel("[bold italic white on black]OS CONTROL[/bold italic white on black] Enabled", box=box.DOUBLE, expand=False), style="white on black")
        # print(">\n\n")
        # console.print(Panel("[bold italic white on black]OS CONTROL[/bold italic white on black] Enabled", box=box.SQUARE, expand=False), style="white on black")

        if not args.offline and not args.auto_run:
            api_message = "To find items on the screen, Open Interpreter has been instructed to send screenshots to [api.openinterpreter.com](https://api.openinterpreter.com/) (we do not store them). Add `--offline` to attempt this locally."
            display_markdown_message(api_message)
            print("")

        if not args.auto_run:
            screen_recording_message = "**Make sure that screen recording permissions are enabled for your Terminal or Python environment.**"
            display_markdown_message(screen_recording_message)
            print("")

        # # FOR TESTING ONLY
        # # Install Open Interpreter from GitHub
        # for chunk in interpreter.computer.run(
        #     "shell",
        #     "pip install git+https://github.com/KillianLucas/open-interpreter.git",
        # ):
        #     if chunk.get("format") != "active_line":
        #         print(chunk.get("content"))

        # Give it access to the computer via Python
        interpreter.computer.run(
            "python",
            "import time\nfrom interpreter import interpreter\ncomputer = interpreter.interpreter.computer",  # We ask it to use time, so
            display=args.verbose,
        )

        if not args.auto_run:
            display_markdown_message(
                "**Warning:** In this mode, Open Interpreter will not require approval before performing actions. Be ready to close your terminal."
            )
            print("")  # < - Aesthetic choice
        args.profile = "os.py"

    if args.local:
        args.profile = "local.py"

    ### Set attributes on interpreter, so that a profile script can read the arguments passed in via the CLI

    set_attributes(args, arguments)

    ### Apply profile

    interpreter = profile(interpreter, args.profile or get_argument_dictionary(arguments, "profile")["default"])

    ### Set attributes on interpreter, because the arguments passed in via the CLI should override profile

    set_attributes(args, arguments)

    ### Set some helpful settings we know are likely to be true

    if interpreter.llm.model == "gpt-4" or interpreter.llm.model == "openai/gpt-4":
        if interpreter.llm.context_window is None:
            interpreter.llm.context_window = 6500
        if interpreter.llm.max_tokens is None:
            interpreter.llm.max_tokens = 4096
        if interpreter.llm.supports_functions is None:
            interpreter.llm.supports_functions = (
                False if "vision" in interpreter.llm.model else True
            )

    elif interpreter.llm.model.startswith("gpt-4") or interpreter.llm.model.startswith(
        "openai/gpt-4"
    ):
        if interpreter.llm.context_window is None:
            interpreter.llm.context_window = 123000
        if interpreter.llm.max_tokens is None:
            interpreter.llm.max_tokens = 4096
        if interpreter.llm.supports_functions is None:
            interpreter.llm.supports_functions = (
                False if "vision" in interpreter.llm.model else True
            )

    if interpreter.llm.model.startswith(
        "gpt-3.5-turbo"
    ) or interpreter.llm.model.startswith("openai/gpt-3.5-turbo"):
        if interpreter.llm.context_window is None:
            interpreter.llm.context_window = 16000
        if interpreter.llm.max_tokens is None:
            interpreter.llm.max_tokens = 4096
        if interpreter.llm.supports_functions is None:
            interpreter.llm.supports_functions = True

    ### Check for update

    try:
        if not interpreter.offline:
            # This message should actually be pushed into the utility
            if check_for_update():
                display_markdown_message(
                    "> **A new version of Open Interpreter is available.**\n>Please run: `pip install --upgrade open-interpreter`\n\n---"
                )
    except:
        # Doesn't matter
        pass

    if interpreter.llm.api_base:
        if (
            not interpreter.llm.model.lower().startswith("openai/")
            and not interpreter.llm.model.lower().startswith("azure/")
            and not interpreter.llm.model.lower().startswith("ollama")
            and not interpreter.llm.model.lower().startswith("jan")
            and not interpreter.llm.model.lower().startswith("local")
        ):
            interpreter.llm.model = "openai/" + interpreter.llm.model
        elif interpreter.llm.model.lower().startswith("jan/"):
            # Strip jan/ from the model name
            interpreter.llm.model = interpreter.llm.model[4:]

    # If --conversations is used, run conversation_navigator
    if args.conversations:
        conversation_navigator(interpreter)
        return

    if args.server:
        interpreter.server()
        return

    validate_llm_settings(interpreter)

    interpreter.in_terminal_interface = True

    interpreter.chat()


def set_attributes(args, arguments):
    for argument_name, argument_value in vars(args).items():
        if argument_value is not None:
            if argument_dictionary := get_argument_dictionary(arguments, argument_name):
                if "attribute" in argument_dictionary:
                    attr_dict = argument_dictionary["attribute"]
                    setattr(attr_dict["object"], attr_dict["attr_name"], argument_value)

                    if args.verbose:
                        print(
                            f"Setting attribute {attr_dict['attr_name']} on {attr_dict['object'].__class__.__name__.lower()} to '{argument_value}'..."
                        )


def get_argument_dictionary(arguments: list[dict], key: str) -> dict:
    if len(argument_dictionary_list := list(filter(lambda x: x["name"] == key, arguments))) > 0:
        return argument_dictionary_list[0]
    return {}


def main():
    interpreter = OpenInterpreter(import_computer_api=True)
    try:
        start_terminal_interface(interpreter)
    except KeyboardInterrupt:
        pass
    finally:
        interpreter.computer.terminate()
