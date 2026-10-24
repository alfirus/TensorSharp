"""Distinct diagnostic tags must neither shadow nor supply qualification events."""
import unittest

from test_glm_causal_route_neighbor import neighbor, request, request_log


class CausalRouteNeighborCpuTagTests(unittest.TestCase):
    def test_actual_cpu_tag_is_ignored_without_supplying_missing_core_credit(self):
        marker = "[GLM-CAUSAL-ROUTE-NEIGHBOR-CPU]"
        cpu = (marker + " outcome=completed p0=1872 rows=1 valid=1 prepare_ns=123 commit_ns=45"
               " slot_checks=750 victim_checks=1498 scope=nested_host_wall_not_cpu_exclusive")
        plain, annotated, cpu_only = [], [], []
        for name,prompt,count in neighbor.QUICK_REQUESTS:
            log = request_log(prompt,count,1)
            with_cpu = "\n".join(line + ("\n" + cpu if neighbor.MARKER in line else "")
                                   for line in log.splitlines())
            # Actual native diagnostic fields share names with core fields, but
            # the exact closing-bracket tag is a separate optional protocol.
            observed, decorated = neighbor.records(log), neighbor.records(with_cpu)
            self.assertEqual(observed,decorated)
            self.assertEqual(decorated["malformed_records"],0)
            plain.append(request(name,prompt,count,log))
            annotated.append(request(name,prompt,count,with_cpu))
            missing = request_log(prompt,count,0) + "\n" + cpu
            cpu_only.append(request(name,prompt,count,missing))
        expected = neighbor.qualification(plain,1)
        self.assertTrue(expected["passed"])
        self.assertEqual(expected,neighbor.qualification(annotated,1))
        self.assertFalse(neighbor.qualification(cpu_only,1)["passed"])
        self.assertTrue(all(not row["causal_route_neighbor"]["records"] for row in cpu_only))


if __name__ == "__main__":
    unittest.main()
