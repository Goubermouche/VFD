#include "pch.h"
#include "ImGuiUtilities.h"

#include <imgui.h>

namespace fe {
	void UI::ShiftCursor(float x, float y)
	{
		const ImVec2 cursor = ImGui::GetCursorPos();
		ImGui::SetCursorPos(ImVec2(cursor.x + x, cursor.y + y));
	}

	void UI::ShiftCursorX(float value)
	{
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + value);
	}

	void UI::ShiftCursorY(float value)
	{
		ImGui::SetCursorPosY(ImGui::GetCursorPosY() + value);
	}

	bool UI::ItemHoverable(const ImRect& bb, ImGuiID id)
	{
		auto g = ImGui::GetCurrentContext();

		if (g->CurrentWindow != g->HoveredWindow) {
			return false;
		}

		if (ImGui::IsMouseHoveringRect(bb.Min, bb.Max)) {
			ImGui::SetHoveredID(id);
			return true;
		}

		return false;
	}

    static void NavApplyItemToResult(ImGuiNavItemData* result)
    {
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = g.CurrentWindow;
        result->Window = window;
        result->ID = g.LastItemData.ID;
        result->FocusScopeId = window->DC.NavFocusScopeIdCurrent;
        result->InFlags = g.LastItemData.InFlags;
        result->RectRel = ImGui::WindowRectAbsToRel(window, g.LastItemData.NavRect);
    }

    static void NavUpdateAnyRequestFlag()
    {
        ImGuiContext& g = *GImGui;
        g.NavAnyRequest = g.NavMoveScoringItems || g.NavInitRequest || (0 && g.NavWindow != NULL);
        if (g.NavAnyRequest)
            IM_ASSERT(g.NavWindow != NULL);
    }

    static void NavMoveRequestResolveWithLastItem(ImGuiNavItemData* result)
    {
        ImGuiContext& g = *GImGui;
        g.NavMoveScoringItems = false; // Ensure request doesn't need more processing
        NavApplyItemToResult(result);
        NavUpdateAnyRequestFlag();
    }

    static void NavClampRectToVisibleAreaForMoveDir(ImGuiDir move_dir, ImRect& r, const ImRect& clip_rect)
    {
        if (move_dir == ImGuiDir_Left || move_dir == ImGuiDir_Right)
        {
            r.Min.y = ImClamp(r.Min.y, clip_rect.Min.y, clip_rect.Max.y);
            r.Max.y = ImClamp(r.Max.y, clip_rect.Min.y, clip_rect.Max.y);
        }
        else // FIXME: PageUp/PageDown are leaving move_dir == None
        {
            r.Min.x = ImClamp(r.Min.x, clip_rect.Min.x, clip_rect.Max.x);
            r.Max.x = ImClamp(r.Max.x, clip_rect.Min.x, clip_rect.Max.x);
        }
    }

    static float inline NavScoreItemDistInterval(float a0, float a1, float b0, float b1)
    {
        if (a1 < b0)
            return a1 - b0;
        if (b1 < a0)
            return a0 - b1;
        return 0.0f;
    }

    static void NavProcessItemForTabbingRequest(ImGuiID id)
    {
        ImGuiContext& g = *GImGui;

        // Always store in NavMoveResultLocal (unlike directional request which uses NavMoveResultOther on sibling/flattened windows)
        ImGuiNavItemData* result = &g.NavMoveResultLocal;
        if (g.NavTabbingDir == +1)
        {
            // Tab Forward or SetKeyboardFocusHere() with >= 0
            if (g.NavTabbingResultFirst.ID == 0)
                NavApplyItemToResult(&g.NavTabbingResultFirst);
            if (--g.NavTabbingCounter == 0)
                NavMoveRequestResolveWithLastItem(result);
            else if (g.NavId == id)
                g.NavTabbingCounter = 1;
        }
        else if (g.NavTabbingDir == -1)
        {
            // Tab Backward
            if (g.NavId == id)
            {
                if (result->ID)
                {
                    g.NavMoveScoringItems = false;
                    NavUpdateAnyRequestFlag();
                }
            }
            else
            {
                NavApplyItemToResult(result);
            }
        }
        else if (g.NavTabbingDir == 0)
        {
            // Tab Init
            if (g.NavTabbingResultFirst.ID == 0)
                NavMoveRequestResolveWithLastItem(&g.NavTabbingResultFirst);
        }
    }

    static bool NavScoreItem(ImGuiNavItemData* result)
    {
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = g.CurrentWindow;
        if (g.NavLayer != window->DC.NavLayerCurrent)
            return false;

        // FIXME: Those are not good variables names
        ImRect cand = g.LastItemData.NavRect;   // Current item nav rectangle
        const ImRect curr = g.NavScoringRect;   // Current modified source rect (NB: we've applied Max.x = Min.x in NavUpdate() to inhibit the effect of having varied item width)
        g.NavScoringDebugCount++;

        // When entering through a NavFlattened border, we consider child window items as fully clipped for scoring
        if (window->ParentWindow == g.NavWindow)
        {
            IM_ASSERT((window->Flags | g.NavWindow->Flags) & ImGuiWindowFlags_NavFlattened);
            if (!window->ClipRect.Overlaps(cand))
                return false;
            cand.ClipWithFull(window->ClipRect); // This allows the scored item to not overlap other candidates in the parent window
        }

        // We perform scoring on items bounding box clipped by the current clipping rectangle on the other axis (clipping on our movement axis would give us equal scores for all clipped items)
        // For example, this ensure that items in one column are not reached when moving vertically from items in another column.
        NavClampRectToVisibleAreaForMoveDir(g.NavMoveClipDir, cand, window->ClipRect);

        // Compute distance between boxes
        // FIXME-NAV: Introducing biases for vertical navigation, needs to be removed.
        float dbx = NavScoreItemDistInterval(cand.Min.x, cand.Max.x, curr.Min.x, curr.Max.x);
        float dby = NavScoreItemDistInterval(ImLerp(cand.Min.y, cand.Max.y, 0.2f), ImLerp(cand.Min.y, cand.Max.y, 0.8f), ImLerp(curr.Min.y, curr.Max.y, 0.2f), ImLerp(curr.Min.y, curr.Max.y, 0.8f)); // Scale down on Y to keep using box-distance for vertically touching items
        if (dby != 0.0f && dbx != 0.0f)
            dbx = (dbx / 1000.0f) + ((dbx > 0.0f) ? +1.0f : -1.0f);
        float dist_box = ImFabs(dbx) + ImFabs(dby);

        // Compute distance between centers (this is off by a factor of 2, but we only compare center distances with each other so it doesn't matter)
        float dcx = (cand.Min.x + cand.Max.x) - (curr.Min.x + curr.Max.x);
        float dcy = (cand.Min.y + cand.Max.y) - (curr.Min.y + curr.Max.y);
        float dist_center = ImFabs(dcx) + ImFabs(dcy); // L1 metric (need this for our connectedness guarantee)

        // Determine which quadrant of 'curr' our candidate item 'cand' lies in based on distance
        ImGuiDir quadrant;
        float dax = 0.0f, day = 0.0f, dist_axial = 0.0f;
        if (dbx != 0.0f || dby != 0.0f)
        {
            // For non-overlapping boxes, use distance between boxes
            dax = dbx;
            day = dby;
            dist_axial = dist_box;
            quadrant = ImGetDirQuadrantFromDelta(dbx, dby);
        }
        else if (dcx != 0.0f || dcy != 0.0f)
        {
            // For overlapping boxes with different centers, use distance between centers
            dax = dcx;
            day = dcy;
            dist_axial = dist_center;
            quadrant = ImGetDirQuadrantFromDelta(dcx, dcy);
        }
        else
        {
            // Degenerate case: two overlapping buttons with same center, break ties arbitrarily (note that LastItemId here is really the _previous_ item order, but it doesn't matter)
            quadrant = (g.LastItemData.ID < g.NavId) ? ImGuiDir_Left : ImGuiDir_Right;
        }

#if IMGUI_DEBUG_NAV_SCORING
        char buf[128];
        if (IsMouseHoveringRect(cand.Min, cand.Max))
        {
            ImFormatString(buf, IM_ARRAYSIZE(buf), "dbox (%.2f,%.2f->%.4f)\ndcen (%.2f,%.2f->%.4f)\nd (%.2f,%.2f->%.4f)\nnav %c, quadrant %c", dbx, dby, dist_box, dcx, dcy, dist_center, dax, day, dist_axial, "WENS"[g.NavMoveDir], "WENS"[quadrant]);
            ImDrawList* draw_list = GetForegroundDrawList(window);
            draw_list->AddRect(curr.Min, curr.Max, IM_COL32(255, 200, 0, 100));
            draw_list->AddRect(cand.Min, cand.Max, IM_COL32(255, 255, 0, 200));
            draw_list->AddRectFilled(cand.Max - ImVec2(4, 4), cand.Max + CalcTextSize(buf) + ImVec2(4, 4), IM_COL32(40, 0, 0, 150));
            draw_list->AddText(cand.Max, ~0U, buf);
        }
        else if (g.IO.KeyCtrl) // Hold to preview score in matching quadrant. Press C to rotate.
        {
            if (quadrant == g.NavMoveDir)
            {
                ImFormatString(buf, IM_ARRAYSIZE(buf), "%.0f/%.0f", dist_box, dist_center);
                ImDrawList* draw_list = GetForegroundDrawList(window);
                draw_list->AddRectFilled(cand.Min, cand.Max, IM_COL32(255, 0, 0, 200));
                draw_list->AddText(cand.Min, IM_COL32(255, 255, 255, 255), buf);
            }
        }
#endif

        // Is it in the quadrant we're interesting in moving to?
        bool new_best = false;
        const ImGuiDir move_dir = g.NavMoveDir;
        if (quadrant == move_dir)
        {
            // Does it beat the current best candidate?
            if (dist_box < result->DistBox)
            {
                result->DistBox = dist_box;
                result->DistCenter = dist_center;
                return true;
            }
            if (dist_box == result->DistBox)
            {
                // Try using distance between center points to break ties
                if (dist_center < result->DistCenter)
                {
                    result->DistCenter = dist_center;
                    new_best = true;
                }
                else if (dist_center == result->DistCenter)
                {
                    // Still tied! we need to be extra-careful to make sure everything gets linked properly. We consistently break ties by symbolically moving "later" items
                    // (with higher index) to the right/downwards by an infinitesimal amount since we the current "best" button already (so it must have a lower index),
                    // this is fairly easy. This rule ensures that all buttons with dx==dy==0 will end up being linked in order of appearance along the x axis.
                    if (((move_dir == ImGuiDir_Up || move_dir == ImGuiDir_Down) ? dby : dbx) < 0.0f) // moving bj to the right/down decreases distance
                        new_best = true;
                }
            }
        }

        // Axial check: if 'curr' has no link at all in some direction and 'cand' lies roughly in that direction, add a tentative link. This will only be kept if no "real" matches
        // are found, so it only augments the graph produced by the above method using extra links. (important, since it doesn't guarantee strong connectedness)
        // This is just to avoid buttons having no links in a particular direction when there's a suitable neighbor. you get good graphs without this too.
        // 2017/09/29: FIXME: This now currently only enabled inside menu bars, ideally we'd disable it everywhere. Menus in particular need to catch failure. For general navigation it feels awkward.
        // Disabling it may lead to disconnected graphs when nodes are very spaced out on different axis. Perhaps consider offering this as an option?
        if (result->DistBox == FLT_MAX && dist_axial < result->DistAxial)  // Check axial match
            if (g.NavLayer == ImGuiNavLayer_Menu && !(g.NavWindow->Flags & ImGuiWindowFlags_ChildMenu))
                if ((move_dir == ImGuiDir_Left && dax < 0.0f) || (move_dir == ImGuiDir_Right && dax > 0.0f) || (move_dir == ImGuiDir_Up && day < 0.0f) || (move_dir == ImGuiDir_Down && day > 0.0f))
                {
                    result->DistAxial = dist_axial;
                    new_best = true;
                }

        return new_best;
    }

    static void NavProcessItem()
    {
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = g.CurrentWindow;
        const ImGuiID id = g.LastItemData.ID;
        const ImRect nav_bb = g.LastItemData.NavRect;
        const ImGuiItemFlags item_flags = g.LastItemData.InFlags;

        // Process Init Request
        if (g.NavInitRequest && g.NavLayer == window->DC.NavLayerCurrent)
        {
            // Even if 'ImGuiItemFlags_NoNavDefaultFocus' is on (typically collapse/close button) we record the first ResultId so they can be used as a fallback
            const bool candidate_for_nav_default_focus = (item_flags & (ImGuiItemFlags_NoNavDefaultFocus | ImGuiItemFlags_Disabled)) == 0;
            if (candidate_for_nav_default_focus || g.NavInitResultId == 0)
            {
                g.NavInitResultId = id;
                g.NavInitResultRectRel = ImGui::WindowRectAbsToRel(window, nav_bb);
            }
            if (candidate_for_nav_default_focus)
            {
                g.NavInitRequest = false; // Found a match, clear request
                NavUpdateAnyRequestFlag();
            }
        }

        // Process Move Request (scoring for navigation)
        // FIXME-NAV: Consider policy for double scoring (scoring from NavScoringRect + scoring from a rect wrapped according to current wrapping policy)
        if (g.NavMoveScoringItems)
        {
            const bool is_tab_stop = (item_flags & ImGuiItemFlags_Inputable) && (item_flags & (ImGuiItemFlags_NoTabStop | ImGuiItemFlags_Disabled)) == 0;
            const bool is_tabbing = (g.NavMoveFlags & ImGuiNavMoveFlags_Tabbing) != 0;
            if (is_tabbing)
            {
                if (is_tab_stop || (g.NavMoveFlags & ImGuiNavMoveFlags_FocusApi))
                    NavProcessItemForTabbingRequest(id);
            }
            else if ((g.NavId != id || (g.NavMoveFlags & ImGuiNavMoveFlags_AllowCurrentNavId)) && !(item_flags & (ImGuiItemFlags_Disabled | ImGuiItemFlags_NoNav)))
            {
                ImGuiNavItemData* result = (window == g.NavWindow) ? &g.NavMoveResultLocal : &g.NavMoveResultOther;
                if (!is_tabbing)
                {
                    if (NavScoreItem(result))
                        NavApplyItemToResult(result);

                    // Features like PageUp/PageDown need to maintain a separate score for the visible set of items.
                    const float VISIBLE_RATIO = 0.70f;
                    if ((g.NavMoveFlags & ImGuiNavMoveFlags_AlsoScoreVisibleSet) && window->ClipRect.Overlaps(nav_bb))
                        if (ImClamp(nav_bb.Max.y, window->ClipRect.Min.y, window->ClipRect.Max.y) - ImClamp(nav_bb.Min.y, window->ClipRect.Min.y, window->ClipRect.Max.y) >= (nav_bb.Max.y - nav_bb.Min.y) * VISIBLE_RATIO)
                            if (NavScoreItem(&g.NavMoveResultLocalVisible))
                                NavApplyItemToResult(&g.NavMoveResultLocalVisible);
                }
            }
        }

        // Update window-relative bounding box of navigated item
        if (g.NavId == id)
        {
            g.NavWindow = window;                                           // Always refresh g.NavWindow, because some operations such as FocusItem() don't have a window.
            g.NavLayer = window->DC.NavLayerCurrent;
            g.NavFocusScopeId = window->DC.NavFocusScopeIdCurrent;
            g.NavIdIsAlive = true;
            window->NavRectRel[window->DC.NavLayerCurrent] = ImGui::WindowRectAbsToRel(window, nav_bb);    // Store item bounding box (relative to window position)
        }
    }

    static bool IsClippedEx(const ImRect& bb, ImGuiID id)
    {
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = g.CurrentWindow;

        if (!bb.Overlaps(window->ClipRect)) {
            if (id == 0 || (id != g.ActiveId && id != g.NavId)) {
                if (!g.LogEnabled) {
                    return true;
                }
            }
        }

        return false;
    }

	bool UI::ItemAdd(const ImRect& bb, ImGuiID id, const ImRect* nav_bb_arg, ImGuiItemFlags extra_flags)
	{
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = g.CurrentWindow;

        // Set item data
        // (DisplayRect is left untouched, made valid when ImGuiItemStatusFlags_HasDisplayRect is set)
        g.LastItemData.ID = id;
        g.LastItemData.Rect = bb;
        g.LastItemData.NavRect = nav_bb_arg ? *nav_bb_arg : bb;
        g.LastItemData.InFlags = g.CurrentItemFlags | extra_flags;
        g.LastItemData.StatusFlags = ImGuiItemStatusFlags_None;

        // Directional navigation processing
        if (id != 0)
        {
            // Runs prior to clipping early-out
            //  (a) So that NavInitRequest can be honored, for newly opened windows to select a default widget
            //  (b) So that we can scroll up/down past clipped items. This adds a small O(N) cost to regular navigation requests
            //      unfortunately, but it is still limited to one window. It may not scale very well for windows with ten of
            //      thousands of item, but at least NavMoveRequest is only set on user interaction, aka maximum once a frame.
            //      We could early out with "if (is_clipped && !g.NavInitRequest) return false;" but when we wouldn't be able
            //      to reach unclipped widgets. This would work if user had explicit scrolling control (e.g. mapped on a stick).
            // We intentionally don't check if g.NavWindow != NULL because g.NavAnyRequest should only be set when it is non null.
            // If we crash on a NULL g.NavWindow we need to fix the bug elsewhere.
            window->DC.NavLayersActiveMaskNext |= (1 << window->DC.NavLayerCurrent);
            if (g.NavId == id || g.NavAnyRequest) {
                if (g.NavWindow->RootWindowForNav == window->RootWindowForNav) {
                    if (window == g.NavWindow || ((window->Flags | g.NavWindow->Flags) & ImGuiWindowFlags_NavFlattened)) {
                        NavProcessItem();
                    }
                }
            }

            // [DEBUG] People keep stumbling on this problem and using "" as identifier in the root of a window instead of "##something".
            // Empty identifier are valid and useful in a small amount of cases, but 99.9% of the time you want to use "##something".
            // READ THE FAQ: https://dearimgui.org/faq
            IM_ASSERT(id != window->ID && "Cannot have an empty ID at the root of a window. If you need an empty label, use ## and read the FAQ about how the ID Stack works!");
        }

        g.NextItemData.Flags = ImGuiNextItemDataFlags_None;

        if (window->DC.CurrentLayoutItem) {
            window->DC.CurrentLayoutItem->MeasuredBounds.Max = ImMax(window->DC.CurrentLayoutItem->MeasuredBounds.Max, bb.Max);
        }

        // Clipping test
        const bool is_clipped = IsClippedEx(bb, id);
        if (is_clipped) {
            return false;
        }

        //if (g.IO.KeyAlt) window->DrawList->AddRect(bb.Min, bb.Max, IM_COL32(255,255,0,120)); // [DEBUG]

        // We need to calculate this now to take account of the current clipping rectangle (as items like Selectable may change them)
        if (ImGui::IsMouseHoveringRect(bb.Min, bb.Max)) {
            g.LastItemData.StatusFlags |= ImGuiItemStatusFlags_HoveredRect;
        }

        return true;
	}
}