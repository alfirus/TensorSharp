import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seed_cache import fetch


class SeedCacheTests(unittest.TestCase):
    def entry(self, data=b'expected archive'):
        return {'fileName': 'package.conda', 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'url': 'https://example.invalid/package.conda'}

    def test_verified_cache_reuse_does_not_contact_network(self):
        data = b'expected archive'
        entry = self.entry(data)
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, entry['sha256'], entry['fileName'])
            target.parent.mkdir()
            target.write_bytes(data)
            with patch('urllib.request.urlopen', side_effect=AssertionError('Network must not be used')):
                self.assertTrue(fetch(entry, root)['cached'])

    def test_corrupt_download_never_becomes_a_cache_entry(self):
        entry = self.entry()
        with tempfile.TemporaryDirectory() as root:
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'incorrect bytes!')):
                with self.assertRaisesRegex(ValueError, 'differs from pinned'):
                    fetch(entry, root)
            self.assertFalse(Path(root, entry['sha256'], entry['fileName']).exists())
            self.assertFalse(list(Path(root).rglob('*.partial')))

    def test_oversized_download_is_rejected_before_publication(self):
        entry = self.entry()
        with tempfile.TemporaryDirectory() as root:
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'x' * (entry['bytes'] + 1))):
                with self.assertRaisesRegex(ValueError, 'exceeds pinned size'):
                    fetch(entry, root)
            self.assertFalse(Path(root, entry['sha256'], entry['fileName']).exists())

    def test_manifest_filename_cannot_escape_the_cache(self):
        entry = self.entry()
        entry['fileName'] = '../escape'
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'Invalid manifest cache path'):
                fetch(entry, root)


if __name__ == '__main__':
    unittest.main()
