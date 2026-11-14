"""Actual dir_fd cleanup plus non-mutating adversarial audit events."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pairwise.runtime import configure_runtime,EXT_ROOT,REPO_ROOT,_check_mutation_path
configure_runtime()


class RuntimeGuardTests(unittest.TestCase):
    def test_temp_directory_nested_cleanup_uses_directory_descriptor(self):
        with tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp') as temporary:
            folder=Path(temporary)
            (folder/'nested').mkdir()
            (folder/'nested'/'inside.txt').write_text('generated entirely inside extension')
            fd=os.open(folder/'nested',os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.chmod('inside.txt',0o600,dir_fd=fd)
                os.utime('inside.txt',ns=(1,1),dir_fd=fd)
                os.rename('inside.txt','renamed.txt',src_dir_fd=fd,dst_dir_fd=fd)
                os.unlink('renamed.txt',dir_fd=fd)
                (folder/'nested'/'cleanup.txt').write_text('shutil must remove this using dir_fd')
            finally:
                os.close(fd)
        self.assertFalse(folder.exists())

    def test_outside_directory_descriptor_and_parent_traversal_stay_blocked(self):
        outer=os.open(REPO_ROOT,os.O_RDONLY|os.O_DIRECTORY)
        inner=os.open(EXT_ROOT,os.O_RDONLY|os.O_DIRECTORY)
        try:
            # The path deliberately does not exist, so even a broken guard could
            # not modify an original. We require the guard's ValueError, not the
            # syscall's FileNotFoundError, to establish rejection occurs first.
            with self.assertRaises(ValueError):
                os.unlink('__v2_guard_nonexistent_outside__',dir_fd=outer)
            with self.assertRaises(ValueError):
                os.unlink('../__v2_guard_nonexistent_outside__',dir_fd=inner)
            # These explicit audit events execute no filesystem operations.
            for event,args in [
                ('os.mkdir',('blocked',0o700,outer)),
                ('os.chmod',('blocked',0o600,outer)),
                ('os.utime',('blocked',None,(1,1),outer)),
                ('os.rename',('inside','outside',inner,outer)),
                ('os.rename',('outside','inside',outer,inner)),
                ('os.link',('inside','outside',inner,outer)),
                ('os.symlink',('source','outside',outer)),
            ]:
                with self.subTest(event=event,args=args), self.assertRaises(ValueError):
                    sys.audit(event,*args)
        finally:
            os.close(outer);os.close(inner)

    def test_absolute_paths_ignore_dir_fd_and_unknown_fd_fails_closed(self):
        outside_fd=os.open(REPO_ROOT,os.O_RDONLY|os.O_DIRECTORY)
        try:
            with tempfile.TemporaryDirectory(dir=EXT_ROOT/'tmp') as temporary:
                folder=Path(temporary)
                inside=folder/'absolute.txt';inside.write_text('safe')
                os.unlink(str(inside),dir_fd=outside_fd)
                # /proc's existing descriptor symlink tests resolution without
                # creating any new link or file outside the extension.
                with self.assertRaises(ValueError):
                    _check_mutation_path(f'/proc/self/fd/{outside_fd}/__nonexistent__')
                with self.assertRaises(ValueError):
                    _check_mutation_path('inside',2**30)
        finally:
            os.close(outside_fd)


if __name__=='__main__':
    unittest.main()
