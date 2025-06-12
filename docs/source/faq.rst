FAQ
===

**Q:** Why isn’t my NEES within the 95% bounds?  
**A:** Check that your process noise `Q` isn’t too small; under-modelling leads to overconfidence.

**Q:** I get a singular matrix in the update step—what now?  
**A:** The filter adds a tiny ridge to `S` and falls back to the pseudo-inverse if needed.

**Q:** How do I freeze some states during update?  
**A:** Use the `SKF` class with an `update_mask`, or go further with the `PSKF` weight vector.
