from abc import ABC, abstractmethod

class Scroll:

    def __init__(self, s_type='', depth=1) -> None:
        self.scroll_type = s_type
        self.depth = depth
        self.render_scroll = [0, 0]

    @abstractmethod
    def scroll(self):
        pass

class Follow(Scroll):
    def __init__(self, s_type='', depth : float=1.0) -> None:
        super().__init__(s_type=s_type, depth=depth)

    def scroll(self, surf, dt, pos=(0, 0), offset=(0, 0)) -> list[int]:
        render_scroll = [0, 0]
        pos = list(pos)

        self.render_scroll[0] += ((pos[0] - offset[0] - surf.get_width() / 2) - self.render_scroll[0]) * self.depth
        self.render_scroll[1] += ((pos[1] - offset[1] - surf.get_height() / 2) - self.render_scroll[1]) * self.depth
        
        render_scroll = [int(self.render_scroll[0]), int(self.render_scroll[1])]

        return render_scroll