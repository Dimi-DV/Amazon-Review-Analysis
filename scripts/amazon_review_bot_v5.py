import json
import time
import random
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

class AmazonReviewScraper:
    def __init__(self, url, executable_path):
        self.url = url
        self.executable_path = executable_path
        self.driver = None
        self.reviews = []

    def setup_driver(self):
        service = Service(executable_path=self.executable_path)
        self.driver = webdriver.Chrome(service=service)

        user_agent = UserAgent().random
        self.driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": user_agent})


    def set_star_filter(self, star):
        if self.driver is None:
            self.setup_driver()

        # Visit the original product page first
        self.driver.get(self.url)
        time.sleep(5)

        star_map = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
        }
        star_string = star_map.get(star, "one")

        try:
            # Find the one_star filter link
            one_star_link_xpath = '//a[@class="a-link-normal 1star"]'
            one_star_link_element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, one_star_link_xpath)))
            one_star_link_element.click()
            time.sleep(5)
        except:
            print("Failed to set the star filter.")

        # Check if the new 1-star filter button is present and click on it
        try:
            new_one_star_button = self.driver.find_element_by_xpath(
                '//span[@class="a-declarative"]/a[contains(@class, "a-link-normal 1star")]')
            new_one_star_button.click()
            time.sleep(5)
        except:
            print("New 1-star filter button not found.")

    def scrape_current_page(self):
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        review_titles = soup.find_all(attrs={"data-hook": "review-title"})
        review_texts = soup.find_all(attrs={"data-hook": "review-body"})

        for title, text in zip(review_titles, review_texts):
            self.reviews.append({'title': title.text.strip(), 'text': text.text.strip()})

    def scrape_reviews(self):
        while True:
            self.scrape_current_page()

            try:
                next_page_xpath = '//ul[@class="a-pagination"]/li[@class="a-last"]/a'
                next_page_element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, next_page_xpath)))
                next_page_element.click()
                time.sleep(random.uniform(5, 10))
            except:
                print("No more pages to scrape.")
                break

    def save_reviews_to_file(self, filename):
        with open(filename, 'w') as json_file:
            json.dump(self.reviews, json_file, ensure_ascii=False, indent=4)

    def run(self, star_rating, output_filename):
        self.setup_driver()
        self.set_star_filter(star_rating)
        self.scrape_reviews()
        self.save_reviews_to_file(output_filename)
        self.quit_driver()

if __name__ == "__main__":
    # Put your amazon link below, make the link as concise as possible

    url = "..."

    # Install the chromedriver if you don't already have one

    scraper = AmazonReviewScraper(url, '/chromedriver')

    # Desired star review to be scraped and desired save destination and filename (1 star and example destination by default)

    scraper.run(1, 'user/data/filename')