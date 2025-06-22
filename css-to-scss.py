#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught
# css-to-scss.py

"""
This script converts CSS content into SCSS format.
It handles:
  1. Parsing flat CSS rules and converting them into a nested structure.
  2. Converting CSS variable definitions (e.g., `--name: value;`) to SCSS variables (e.g., `$name: value;`).
  3. Formatting the nested structure into a readable SCSS string.
It is designed to be run as a command-line script, taking an input CSS file and producing an output SCSS file.
"""

import re
import sys
import os
from typing import List, Dict, Union, Any


def parse_css_rules_flat(
    css_content: str,
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Parses CSS content into a flat list of rules, each containing its selector
    and a list of its properties.
    It first removes CSS comments to simplify parsing.
    """
    # Remove CSS comments (single-line // or multi-line /* ... */)
    # Note: SCSS comments are //, CSS comments are /* */. We only need to remove /* */.
    css_content = re.sub(r"/\*[\s\S]*?\*/", "", css_content)

    # Use a regular expression to find all selector { ... } blocks.
    # re.DOTALL makes '.' match newlines, important for multi-line properties.
    # The regex is designed to capture the selector part (non-greedy until '{')
    # and the content inside the braces (non-greedy until '}').
    rule_blocks = re.findall(r"([^{]+?)\s*\{([^}]+?)\}", css_content, re.DOTALL)

    flat_rules = []
    for selector_raw, properties_raw in rule_blocks:
        selector = selector_raw.strip()
        # Split properties by semicolon, strip whitespace, and filter out empty strings.
        properties = [
            prop.strip() for prop in properties_raw.split(";") if prop.strip()
        ]
        flat_rules.append({"selector": selector, "properties": properties})
    return flat_rules


def build_nested_structure(
    flat_rules: List[Dict[str, Union[str, List[str]]]],
) -> List[Dict[str, Any]]:
    """
    Builds a nested SCSS-like structure from a flat list of CSS rules.
    This function applies a heuristic to identify parent-child relationships
    and pseudo-class/element nesting.
    """
    # Initialize a dictionary to map selectors to their rule objects.
    # This allows for quick lookup and modification of rules.
    # '_is_nested' flag tracks if a rule has been moved inside another rule.
    all_rules_map: Dict[
        str, Dict[str, Union[str, List[str], List[Dict[str, Any]], bool]]
    ] = {
        rule["selector"]: {
            "selector": rule["selector"],
            "properties": list(rule["properties"]),  # Copy properties list
            "children": [],
            "_is_nested": False,
        }
        for rule in flat_rules
    }

    # Iterate through the selectors to find potential parent-child relationships.
    # We iterate on a copy of keys because the original map might be modified.
    selectors_to_process: List[str] = list(all_rules_map.keys())

    # Build the nested structure iteratively. This approach is O(N^2) in the
    # worst case (where N is the number of rules), but practical for typical CSS.
    for i, current_selector in enumerate(selectors_to_process):
        # Skip if the current rule has already been nested under another parent.
        if (
            current_selector not in all_rules_map
            or all_rules_map[current_selector]["_is_nested"]
        ):
            continue

        current_rule_obj = all_rules_map[current_selector]

        # Iterate through all other selectors to find potential children of the current_rule_obj.
        for j, potential_child_selector in enumerate(selectors_to_process):
            if i == j:
                continue  # Don't compare a rule with itself.

            # Skip if the potential child rule has already been processed or nested.
            if (
                potential_child_selector not in all_rules_map
                or all_rules_map[potential_child_selector]["_is_nested"]
            ):
                continue

            # Heuristic Rule 1: Descendant selector (e.g., '.parent .child' -> '.parent { .child { ... } }')
            # Check if the potential child selector starts with the current selector followed by a space.
            if potential_child_selector.startswith(current_selector + " "):
                # Extract the relative child selector part (e.g., '.child').
                child_selector_part = potential_child_selector[
                    len(current_selector) :
                ].strip()

                # Get the actual child rule object from the map.
                child_block = all_rules_map[potential_child_selector]
                child_block["selector"] = (
                    child_selector_part  # Update selector to be relative in SCSS.
                )
                child_block["_is_nested"] = (
                    True  # Mark as nested so it's not a top-level rule.
                )
                current_rule_obj["children"].append(child_block)  # Add as a child.

            # Heuristic Rule 2: Pseudo-class/element or combined class/ID (e.g., '.button:hover', '.button.active')
            # Check if the potential child selector starts exactly with the current selector
            # AND has a suffix that starts with ':' (pseudo) or '.' (another class) or '#' (ID).
            elif potential_child_selector.startswith(current_selector):
                suffix = potential_child_selector[len(current_selector) :].strip()
                if suffix and (
                    suffix.startswith(":")
                    or suffix.startswith(".")
                    or suffix.startswith("#")
                ):
                    # Prepend '&' to make it a direct nested selector in SCSS (e.g., '&:hover').
                    child_selector_part = "&" + suffix
                    child_block = all_rules_map[potential_child_selector]
                    child_block["selector"] = child_selector_part
                    child_block["_is_nested"] = True
                    current_rule_obj["children"].append(child_block)

    # Collect all rules that were not marked as nested (i.e., they remain top-level rules).
    final_nested_rules = [
        rule for _, rule in all_rules_map.items() if not rule["_is_nested"]
    ]

    # Note: This heuristic handles one level of nesting based on direct parent-child
    # selector patterns. For arbitrarily deep nesting (e.g., .grandparent .parent .child),
    # a recursive application of `build_nested_structure` to children would be needed,
    # which is more complex with this flat parsing approach.
    return final_nested_rules


def format_scss(rules: List[Dict[str, Any]], indent_level=0) -> str:
    """
    Formats the structured rules (after nesting) into a readable SCSS string,
    applying proper indentation.
    It also performs the `var(--name)` to `$name` conversion within properties.
    """
    scss_output_lines: List[Union[str, List[str]]] = []
    indent_str = "  " * indent_level  # Use two spaces for indentation

    for rule in rules:
        (selector, properties, children) = (
            rule["selector"],
            rule["properties"],
            rule["children"],
        )

        # Start of the rule block (e.g., `.selector {`)
        scss_output_lines.append(f"{indent_str}{selector} {{")

        # Add properties to the current rule.
        for prop in properties:
            # Convert `var(--name)` in property values to `$name`.
            prop = re.sub(r"var\(--([a-zA-Z0-9\-_]+)\)", r"$\1", prop)
            scss_output_lines.append(
                f"{indent_str}  {prop};"
            )  # Add semicolon for SCSS properties

        # If there are nested child rules, format them recursively.
        if children:
            # Add a newline for better readability between properties and nested rules.
            if properties:
                scss_output_lines.append("")
            # Recursively call format_scss for children, increasing indent level.
            scss_output_lines.append(format_scss(children, indent_level + 1))

        # End of the rule block (e.g., `}`)
        scss_output_lines.append(f"{indent_str}}}")
        # Add an extra newline between top-level rules for better visual separation.
        if indent_level == 0:
            scss_output_lines.append("")

    return "\n".join(scss_output_lines)


def convert_css_to_scss_main(
    css_content: str,
) -> str:
    """
    Main function to orchestrate the CSS to SCSS conversion.
    1. Converts CSS variable definitions (`--name: value;` to `$name: value;`).
    2. Parses flat CSS rules.
    3. Builds a nested SCSS structure based on heuristics.
    4. Formats the nested structure into the final SCSS string.
    """
    # Step 1: Convert CSS variable definitions (`--name: value;` to `$name: value;`)
    # This is done early to ensure variable definitions themselves are converted.
    css_content_with_scss_vars = re.sub(r"--([a-zA-Z0-9\-_]+):", r"$\1:", css_content)

    # Step 2: Parse the CSS content into a flat list of rules.
    flat_rules = parse_css_rules_flat(css_content_with_scss_vars)

    # Step 3: Build the nested SCSS structure from the flat rules.
    nested_structure = build_nested_structure(flat_rules)

    # Step 4: Format the nested structure into the final SCSS string.
    # This step also handles `var(--name)` to `$name` conversion in property values.
    return format_scss(nested_structure)


# Example Usage (as a command-line script)
if __name__ == "__main__":
    # Check if correct number of arguments are provided (input_file, output_file)
    if len(sys.argv) != 3:
        print("Usage: python css_to_scss.py <input_css_file> <output_scss_file>")
        print("Example: python css_to_scss.py style.css style.scss")
        sys.exit(1)

    (input_file, output_file) = (sys.argv[1], sys.argv[2])

    # Validate if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    try:
        # Read the content of the input CSS file
        with open(input_file, "r", encoding="utf-8") as f:
            css_data = f.read()

        # Perform the conversion
        scss_data = convert_css_to_scss_main(css_data)

        # Write the resulting SCSS content to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(scss_data)

        print(f"Successfully converted '{input_file}' to '{output_file}'")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)
