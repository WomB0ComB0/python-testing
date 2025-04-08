#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=all

"""
LinkedIn Profile Scraper

This script automates the process of scraping LinkedIn profiles using the linkedin_scraper library.
It handles authentication, navigates to a specified profile, and extracts structured information
including personal details, work experiences, education, interests, and accomplishments.

The script includes several patches to handle common exceptions that occur due to LinkedIn's
changing structure and anti-scraping measures.

Requirements:
- Python 3.7+
- linkedin_scraper library
- Selenium WebDriver
- Chrome WebDriver executable
- Valid LinkedIn credentials stored in a .env file

Environment Variables:
- LINKEDIN_USER: Your LinkedIn username/email
- LINKEDIN_PASSWORD: Your LinkedIn password

Usage:
    $ python linkedin.py

Output:
    A dictionary containing structured profile data printed to stdout.
"""

import os
from linkedin_scraper import Person, actions
from linkedin_scraper.objects import Experience
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
from pathlib import Path
import asyncio
from selenium.common.exceptions import (
    NoAlertPresentException,
    StaleElementReferenceException,
    TimeoutException,
)

# Load environment variables from .env file in the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# Initialize Chrome WebDriver
service = Service(executable_path="/usr/bin/chromedriver")
driver = webdriver.Chrome(service=service)

# Get LinkedIn credentials from environment variables
try:
    email = os.getenv("LINKEDIN_USER")
    password = os.getenv("LINKEDIN_PASSWORD")
except Exception as e:
    print(f"Error getting email or password: {e}")
    exit(1)


async def main() -> None:
    """
    Main function that handles the LinkedIn profile scraping process.
    
    This function:
    1. Logs into LinkedIn using provided credentials
    2. Applies patches to handle common exceptions
    3. Scrapes a specific LinkedIn profile
    4. Formats and prints the extracted data
    5. Ensures the browser is closed properly
    
    Returns:
        None
    """
    # Login to LinkedIn
    try:
        actions.login(driver, email, password)
    except Exception as e:
        print(f"Error logging in: {e}")
        exit(1)

    # Patch the focus method to handle NoAlertPresentException
    original_focus = Person.focus

    def patched_focus(self):
        """
        Patched version of Person.focus that handles NoAlertPresentException.
        
        This patch prevents the script from crashing when LinkedIn doesn't show
        an expected alert dialog.
        """
        try:
            original_focus(self)
        except NoAlertPresentException:
            pass

    Person.focus = patched_focus

    # Patch the get_experiences method to handle the "too many values to unpack" error
    original_get_experiences = Person.get_experiences

    def patched_get_experiences(self):
        """
        Patched version of Person.get_experiences that handles common exceptions.
        
        This patch addresses:
        1. "Too many values to unpack" errors caused by LinkedIn structure changes
        2. StaleElementReferenceException when elements are no longer attached to the DOM
        3. TimeoutException when elements take too long to load
        
        Returns:
            list: List of Experience objects, or a fallback placeholder if extraction fails
        """
        try:
            return original_get_experiences(self)
        except ValueError as e:
            if "too many values to unpack" in str(e):
                print(
                    "LinkedIn structure has changed. Using fallback method for experiences."
                )
                # Simplified fallback implementation
                self.add_experience(
                    Experience(
                        institution_name="Unable to parse due to LinkedIn changes",
                        position_title="See profile for details",
                        from_date="",
                        to_date="",
                        duration="",
                        location="",
                        description="",
                    )
                )
                return self.experiences
            else:
                raise e
        except (StaleElementReferenceException, TimeoutException):
            print(
                "Encountered stale element or timeout. Using fallback for experiences."
            )
            return self.experiences

    Person.get_experiences = patched_get_experiences

    # Scrape the LinkedIn profile
    try:
        # Replace <...> with the actual LinkedIn username to scrape
        person = Person("https://www.linkedin.com/in/<...>", driver=driver)

        # Format and print the extracted profile data
        print(
            {
                "name": person.name,                      # Full name of the person
                "location": person.location,              # Geographic location
                "about": person.about,                    # About/summary section
                "job_title": person.job_title,            # Current job title
                "company": person.company,                # Current company
                "open_to_work": person.open_to_work,      # Whether they're open to work
                "experiences": [                          # List of work experiences
                    {
                        "title": exp.position_title,      # Job title
                        "company": exp.institution_name,  # Company name
                        "from_date": exp.from_date,       # Start date
                        "to_date": exp.to_date,           # End date
                        "duration": exp.duration,         # Duration at position
                        "location": exp.location,         # Job location
                        "description": exp.description,   # Job description
                    }
                    for exp in person.experiences
                ],
                "educations": [                           # List of education entries
                    {
                        "institution": edu.institution_name,  # School/university name
                        "degree": edu.degree,                 # Degree obtained
                        "from_date": edu.from_date,           # Start date
                        "to_date": edu.to_date,               # End date
                        "description": edu.description,       # Education description
                    }
                    for edu in person.educations
                ],
                "interests": [interest.name for interest in person.interests],  # List of interests
                "accomplishments": [                      # List of accomplishments
                    {"category": acc.category, "title": acc.title}
                    for acc in person.accomplishments
                ],
            }
        )
    except (
        NoAlertPresentException,
        StaleElementReferenceException,
        TimeoutException,
    ) as e:
        print(f"Error scraping profile: {e}")
    finally:
        # Ensure the browser is closed properly
        driver.quit()


if __name__ == "__main__":
    asyncio.run(main())
