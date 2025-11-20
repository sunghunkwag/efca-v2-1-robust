from playwright.sync_api import sync_playwright

class BrowserController:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False) # Or True for headless
        self.page = self.browser.new_page()

    def navigate(self, url):
        try:
            self.page.goto(url)
            return f"Successfully navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    def get_content(self):
        try:
            content = self.page.content()
            return content
        except Exception as e:
            return f"Error getting content: {e}"

    def close(self):
        self.browser.close()
        self.playwright.stop()

# Example usage:
if __name__ == '__main__':
    controller = BrowserController()
    print(controller.navigate('https://www.example.com'))
    print("Page content length:", len(controller.get_content()))
    controller.close()
