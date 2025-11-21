import unittest
from unittest.mock import MagicMock, patch
from efca.browser_interface import BrowserController


class TestBrowserInterface(unittest.TestCase):
    """
    Unit tests for the Browser Interface.
    
    Note: These tests use mocking to avoid requiring an actual browser instance.
    """
    
    @patch('efca.browser_interface.sync_playwright')
    def test_initialization(self, mock_playwright):
        """Tests that BrowserController can be initialized."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController(headless=True, default_timeout=5000)
        
        self.assertIsInstance(controller, BrowserController)
        self.assertEqual(controller.default_timeout, 5000)
        mock_pw.chromium.launch.assert_called_once_with(headless=True)
    
    @patch('efca.browser_interface.sync_playwright')
    def test_navigate(self, mock_playwright):
        """Tests navigation method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController()
        result = controller.navigate('https://example.com')
        
        mock_page.goto.assert_called_once()
        self.assertIn('Successfully', result)
    
    @patch('efca.browser_interface.sync_playwright')
    def test_click(self, mock_playwright):
        """Tests click method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController()
        result = controller.click('button#submit')
        
        mock_page.click.assert_called_once()
        self.assertIn('Successfully clicked', result)
    
    @patch('efca.browser_interface.sync_playwright')
    def test_type(self, mock_playwright):
        """Tests type method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController()
        result = controller.type('input#username', 'testuser')
        
        mock_page.fill.assert_called_once_with('input#username', 'testuser')
        self.assertIn('Successfully typed', result)
    
    @patch('efca.browser_interface.sync_playwright')
    def test_wait_for_selector(self, mock_playwright):
        """Tests wait_for_selector method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController()
        result = controller.wait_for_selector('div.content', state='visible')
        
        mock_page.wait_for_selector.assert_called_once()
        self.assertIn('visible', result)
    
    @patch('efca.browser_interface.sync_playwright')
    def test_is_visible(self, mock_playwright):
        """Tests is_visible method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_page.is_visible.return_value = True
        
        controller = BrowserController()
        result = controller.is_visible('button#submit')
        
        self.assertTrue(result)
        mock_page.is_visible.assert_called_once_with('button#submit')
    
    @patch('efca.browser_interface.sync_playwright')
    def test_press_key(self, mock_playwright):
        """Tests press_key method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController()
        result = controller.press_key('Enter')
        
        mock_page.keyboard.press.assert_called_once_with('Enter')
        self.assertIn('Successfully pressed', result)
    
    @patch('efca.browser_interface.sync_playwright')
    def test_close(self, mock_playwright):
        """Tests close method."""
        # Setup mocks
        mock_pw = MagicMock()
        mock_playwright.return_value.start.return_value = mock_pw
        mock_browser = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_context = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        
        controller = BrowserController()
        controller.close()
        
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
