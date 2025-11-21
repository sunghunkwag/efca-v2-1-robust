from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Optional, Union
import time

class BrowserController:
    def __init__(self, headless: bool = False, default_timeout: int = 30000):
        """
        Initialize Browser Controller.
        
        Args:
            headless (bool): Run browser in headless mode
            default_timeout (int): Default timeout in milliseconds
        """
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.set_default_timeout(default_timeout)
        self.default_timeout = default_timeout

    def navigate(self, url: str, wait_until: str = 'domcontentloaded') -> str:
        """
        Navigate to URL with configurable wait condition.
        
        Args:
            url (str): URL to navigate to
            wait_until (str): When to consider navigation succeeded
                            ('load', 'domcontentloaded', 'networkidle')
        """
        try:
            self.page.goto(url, wait_until=wait_until)
            return f"Successfully navigated to {url}"
        except PlaywrightTimeoutError:
            return f"Timeout navigating to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    def get_content(self):
        try:
            content = self.page.content()
            return content
        except Exception as e:
            return f"Error getting content: {e}"

    def click(self, selector: str, timeout: Optional[int] = None) -> str:
        """
        Click an element with optional timeout.
        
        Args:
            selector (str): CSS selector for element
            timeout (int): Optional timeout in milliseconds
        """
        try:
            self.page.click(selector, timeout=timeout or self.default_timeout)
            return f"Successfully clicked {selector}"
        except PlaywrightTimeoutError:
            return f"Timeout waiting for {selector}"
        except Exception as e:
            return f"Error clicking {selector}: {e}"

    def type(self, selector: str, text: str, delay: int = 0) -> str:
        """
        Type text into an element with optional delay between keystrokes.
        
        Args:
            selector (str): CSS selector for input element
            text (str): Text to type
            delay (int): Delay in milliseconds between keystrokes
        """
        try:
            self.page.fill(selector, text)
            if delay > 0:
                self.page.type(selector, text, delay=delay)
            return f"Successfully typed into {selector}"
        except Exception as e:
            return f"Error typing into {selector}: {e}"

    def screenshot(self, path):
        try:
            self.page.screenshot(path=path)
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"Error saving screenshot: {e}"

    def get_title(self) -> str:
        """Get current page title."""
        try:
            return self.page.title()
        except Exception as e:
            return f"Error getting title: {e}"
    
    def wait_for_selector(self, selector: str, timeout: Optional[int] = None, state: str = 'visible') -> str:
        """
        Wait for a selector to be in a specific state.
        
        Args:
            selector (str): CSS selector to wait for
            timeout (int): Optional timeout in milliseconds
            state (str): State to wait for ('attached', 'detached', 'visible', 'hidden')
        """
        try:
            self.page.wait_for_selector(selector, timeout=timeout or self.default_timeout, state=state)
            return f"Element {selector} is {state}"
        except PlaywrightTimeoutError:
            return f"Timeout waiting for {selector} to be {state}"
        except Exception as e:
            return f"Error waiting for {selector}: {e}"
    
    def is_visible(self, selector: str) -> bool:
        """
        Check if an element is visible.
        
        Args:
            selector (str): CSS selector for element
            
        Returns:
            bool: True if element exists and is visible
        """
        try:
            return self.page.is_visible(selector)
        except Exception:
            return False
    
    def press_key(self, key: str) -> str:
        """
        Press a keyboard key.
        
        Args:
            key (str): Key to press (e.g., 'Enter', 'Escape', 'ArrowDown')
        """
        try:
            self.page.keyboard.press(key)
            return f"Successfully pressed {key}"
        except Exception as e:
            return f"Error pressing {key}: {e}"
    
    def type_text(self, text: str, delay: int = 0) -> str:
        """
        Type text directly without targeting a selector.
        
        Args:
            text (str): Text to type
            delay (int): Delay in milliseconds between keystrokes
        """
        try:
            self.page.keyboard.type(text, delay=delay)
            return f"Successfully typed text"
        except Exception as e:
            return f"Error typing text: {e}"
    
    def get_element_text(self, selector: str) -> Optional[str]:
        """
        Get text content of an element.
        
        Args:
            selector (str): CSS selector for element
            
        Returns:
            Optional[str]: Text content or None if error
        """
        try:
            return self.page.text_content(selector)
        except Exception:
            return None
    
    def get_element_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """
        Get an attribute value from an element.
        
        Args:
            selector (str): CSS selector for element
            attribute (str): Attribute name
            
        Returns:
            Optional[str]: Attribute value or None if error
        """
        try:
            return self.page.get_attribute(selector, attribute)
        except Exception:
            return None

    def close(self):
        """Close browser and cleanup resources."""
        try:
            self.context.close()
            self.browser.close()
            self.playwright.stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Example usage:
if __name__ == '__main__':
    controller = BrowserController()
    print(controller.navigate('https://www.example.com'))
    print("Page content length:", len(controller.get_content()))
    print(controller.get_title())
    controller.close()
