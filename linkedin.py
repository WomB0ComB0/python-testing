#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=all

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

load_dotenv(Path(__file__).parent / ".env")

service = Service(executable_path="/usr/bin/chromedriver")
driver = webdriver.Chrome(service=service)

try:
    email = os.getenv("LINKEDIN_USER")
    password = os.getenv("LINKEDIN_PASSWORD")
except Exception as e:
    print(f"Error getting email or password: {e}")
    exit(1)


async def main() -> None:
    try:
        actions.login(driver, email, password)
    except Exception as e:
        print(f"Error logging in: {e}")
        exit(1)

    # Patch the focus method to handle NoAlertPresentException
    original_focus = Person.focus

    def patched_focus(self):
        try:
            original_focus(self)
        except NoAlertPresentException:
            pass

    Person.focus = patched_focus

    # Patch the get_experiences method to handle the "too many values to unpack" error
    original_get_experiences = Person.get_experiences

    def patched_get_experiences(self):
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

    try:
        person = Person("https://www.linkedin.com/in/<...>", driver=driver)

        print(
            {
                "name": person.name,
                "location": person.location,
                "about": person.about,
                "job_title": person.job_title,
                "company": person.company,
                "open_to_work": person.open_to_work,
                "experiences": [
                    {
                        "title": exp.position_title,
                        "company": exp.institution_name,
                        "from_date": exp.from_date,
                        "to_date": exp.to_date,
                        "duration": exp.duration,
                        "location": exp.location,
                        "description": exp.description,
                    }
                    for exp in person.experiences
                ],
                "educations": [
                    {
                        "institution": edu.institution_name,
                        "degree": edu.degree,
                        "from_date": edu.from_date,
                        "to_date": edu.to_date,
                        "description": edu.description,
                    }
                    for edu in person.educations
                ],
                "interests": [interest.name for interest in person.interests],
                "accomplishments": [
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
        driver.quit()


if __name__ == "__main__":
    asyncio.run(main())
