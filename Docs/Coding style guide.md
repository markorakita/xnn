- Classes, namespaces and methods should use UpperCamelCase.
- Parameters and variables use lowerCamelCase.
- Use spaces instead of tabs.
- Braces are required, even for single line statements.
- No magic numbers/strings in code!
- Put space between 'if', 'for', 'while', etc.. and parenthesis. Also between: "a, b", "a + b = c",...
- Constants should have prefix "c_", members "m_", global variables "g_", static variables "s_".
- #define macros should be in all caps.
- Interface classes should have prefix 'I'.
- Use proper grammar in comments (and code in general).
- Write comment headers for each function declaration inside header files, and write proper info.
- Do not put more than one statement or declare more than one variable in the single line.
- Private members and methods come first, then protected, and then public.
- Sort includes in this order:
    1) If .cpp/.cu file then we first include header of this file.
    2) C system headers
    3) C++ standard library headers
    4) Imported libraries' headers
    5) This project headers